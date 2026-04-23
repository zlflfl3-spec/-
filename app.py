import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from io import BytesIO

# ═══════════════════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="하우징 그룹핑 분석 도구",
    page_icon="🔌",
    layout="wide"
)

# 한글 폰트 설정
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

st.title("🔌 하우징 그룹핑 분석 도구")
st.caption("엑셀 매트릭스 → Louvain 커뮤니티 탐지 → 그룹 산출")

# ═══════════════════════════════════════════════════════════════════
# 사이드바: 파라미터 설정
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 분석 파라미터")

    st.subheader("1️⃣ 크기 제약")
    max_group_size = st.slider(
        "MAX_GROUP_SIZE (최대 그룹 크기)",
        min_value=5, max_value=100, value=40, step=1,
        help="이 값보다 큰 그룹은 재분할됩니다"
    )
    min_group_size = st.slider(
        "MIN_GROUP_SIZE (최소 그룹 크기)",
        min_value=1, max_value=10, value=2, step=1,
        help="이 값 이하 그룹은 다른 그룹에 흡수됩니다"
    )

    st.subheader("2️⃣ 독립성 하한선")
    independence_min = st.slider(
        "INDEPENDENCE_MIN (%)",
        min_value=0.30, max_value=0.95, value=0.60, step=0.05,
        help="독립도가 이 값 이상인 해상도 중에서 선택"
    )

    st.subheader("3️⃣ Louvain 해상도 범위")
    col_a, col_b = st.columns(2)
    with col_a:
        res_min = st.number_input("최소값", 0.1, 10.0, 0.3, 0.1)
    with col_b:
        res_max = st.number_input("최대값", 0.1, 10.0, 2.0, 0.1)
    res_steps = st.slider("단계 수", 3, 20, 8, 1)

    resolutions = list(np.round(np.linspace(res_min, res_max, res_steps), 2))
    st.caption(f"시도할 해상도: {resolutions}")

    st.markdown("---")
    st.caption("💡 해상도 ↑ = 더 잘게 쪼갬")

# ═══════════════════════════════════════════════════════════════════
# 유틸 함수
# ═══════════════════════════════════════════════════════════════════
def load_matrix(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    common = df.index.intersection(df.columns)
    df = df.loc[common, common]
    M = df.values.astype(float)
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0)
    return pd.DataFrame(M, index=df.index, columns=df.columns)

def build_graph(C):
    G = nx.Graph()
    nodes = list(C.index)
    G.add_nodes_from(nodes)
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            w = C.iloc[i, j]
            if w > 0:
                G.add_edge(nodes[i], nodes[j], weight=float(w))
    return G

def compute_independence(C, partition):
    groups = {}
    for node, gid in partition.items():
        groups.setdefault(gid, []).append(node)
    total_int, total_ext = 0.0, 0.0
    for gid, members in groups.items():
        sub = C.loc[members, members].values
        internal = sub.sum() / 2
        external = C.loc[members, :].values.sum() - sub.sum()
        total_int += internal
        total_ext += external
    if total_int + total_ext == 0:
        return 0
    return total_int / (total_int + total_ext)

def group_stats(C, partition):
    groups = {}
    for node, gid in partition.items():
        groups.setdefault(gid, []).append(node)
    rows = []
    for gid, members in groups.items():
        sub = C.loc[members, members].values
        internal = sub.sum() / 2
        external = C.loc[members, :].values.sum() - sub.sum()
        indep = internal / (internal + external) * 100 if (internal + external) > 0 else 100
        rows.append({
            "그룹ID": gid,
            "크기": len(members),
            "내부연결": int(internal),
            "외부연결": int(external),
            "독립도(%)": round(indep, 1),
            "멤버": ", ".join(sorted(members))
        })
    return pd.DataFrame(rows).sort_values("크기", ascending=False).reset_index(drop=True)

def select_best_resolution(C_act, resolutions, indep_min, log):
    G = build_graph(C_act)
    candidates = []
    scan_rows = []
    for res in resolutions:
        try:
            part = community_louvain.best_partition(G, resolution=res, random_state=42, weight="weight")
        except Exception as e:
            log.append(f"  ⚠ res={res}: {e}")
            continue
        indep = compute_independence(C_act, part)
        groups = {}
        for n_, g_ in part.items():
            groups.setdefault(g_, []).append(n_)
        internal_sum = 0
        for gid, members in groups.items():
            sub = C_act.loc[members, members].values
            internal_sum += sub.sum() / 2
        passed = indep >= indep_min
        scan_rows.append({
            "resolution": res,
            "그룹수": len(groups),
            "내부합": int(internal_sum),
            "독립도": round(indep * 100, 1),
            "통과": "✓" if passed else "✗"
        })
        if passed:
            candidates.append((res, part, internal_sum, indep))
    scan_df = pd.DataFrame(scan_rows)
    if candidates:
        best = max(candidates, key=lambda x: x[2])
        log.append(f"✓ 선택: resolution={best[0]}, 내부합={int(best[2])}, 독립도={best[3]*100:.1f}%")
        return best[1], best[0], scan_df
    else:
        log.append("⚠ 통과 후보 없음 → 독립도 최대 fallback")
        all_results = []
        for res in resolutions:
            try:
                part = community_louvain.best_partition(G, resolution=res, random_state=42, weight="weight")
                indep = compute_independence(C_act, part)
                all_results.append((res, part, indep))
            except:
                pass
        best = max(all_results, key=lambda x: x[2])
        return best[1], best[0], scan_df

def enforce_max_size(C, partition, max_size, log, max_iter=10):
    for iteration in range(max_iter):
        groups = {}
        for node, gid in partition.items():
            groups.setdefault(gid, []).append(node)
        oversized = {g: m for g, m in groups.items() if len(m) > max_size}
        if not oversized:
            break
        new_gid = max(partition.values()) + 1
        for gid, members in oversized.items():
            subC = C.loc[members, members]
            subG = build_graph(subC)
            try:
                sub_part = community_louvain.best_partition(subG, resolution=1.5, random_state=42, weight="weight")
            except:
                continue
            sub_groups = set(sub_part.values())
            if len(sub_groups) <= 1:
                log.append(f"  ⚠ 그룹 {gid}(크기 {len(members)}) 재분할 실패")
                continue
            for node, sub_gid in sub_part.items():
                partition[node] = new_gid + sub_gid
            new_gid += len(sub_groups)
            log.append(f"  ✓ 그룹 {gid}(크기 {len(members)}) → {len(sub_groups)}개 분할")
    return partition

def absorb_small_groups(C, partition, min_size, log, max_iter=20):
    for iteration in range(max_iter):
        groups = {}
        for node, gid in partition.items():
            groups.setdefault(gid, []).append(node)
        small = {g: m for g, m in groups.items() if len(m) <= min_size}
        if not small:
            break
        changed = False
        for gid, members in small.items():
            best_target = None
            best_conn = 0
            for other_gid, other_members in groups.items():
                if other_gid == gid or len(other_members) <= min_size:
                    continue
                conn = C.loc[members, other_members].values.sum()
                if conn > best_conn:
                    best_conn = conn
                    best_target = other_gid
            if best_target is not None:
                for m in members:
                    partition[m] = best_target
                log.append(f"  ✓ 소그룹 크기{len(members)} → 그룹{best_target}로 흡수 (연결={int(best_conn)})")
                changed = True
        if not changed:
            break
    return partition

def renumber_groups(partition, isolated_nodes=None):
    groups = {}
    for node, gid in partition.items():
        groups.setdefault(gid, []).append(node)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    new_part = {}
    for new_id, (old_id, members) in enumerate(sorted_groups):
        for m in members:
            new_part[m] = f"G{new_id:02d}"
    if isolated_nodes:
        for n in isolated_nodes:
            new_part[n] = "G_ISOLATED"
    return new_part

def make_heatmap(C, partition, sort_by_group=True):
    if sort_by_group:
        order = sorted(partition.keys(), key=lambda x: (partition[x], x))
    else:
        order = list(C.index)
    C_sorted = C.loc[order, order]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(C_sorted, cmap="YlOrRd", ax=ax, cbar_kws={"label": "연결 수"},
                xticklabels=True, yticklabels=True)
    ax.set_title("하우징 연결 매트릭스 (그룹순 정렬)", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    return fig

def to_excel_bytes(group_df, C, partition):
    output = BytesIO()
    assignment_df = pd.DataFrame([
        {"하우징": n, "그룹ID": partition[n]} for n in sorted(partition.keys())
    ])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        group_df.to_excel(writer, sheet_name="그룹요약", index=False)
        assignment_df.to_excel(writer, sheet_name="하우징별그룹", index=False)
        C.to_excel(writer, sheet_name="원본매트릭스")
    return output.getvalue()

# ═══════════════════════════════════════════════════════════════════
# 메인 화면
# ═══════════════════════════════════════════════════════════════════
st.header("📁 1. 입력 파일 업로드")
uploaded = st.file_uploader("엑셀 파일 선택 (.xlsx)", type=["xlsx"])

if uploaded:
    try:
        C = load_matrix(uploaded)
        st.success(f"✅ 로드 완료: {C.shape[0]}개 하우징")
        with st.expander("📊 데이터 미리보기"):
            st.dataframe(C.iloc[:10, :10])
    except Exception as e:
        st.error(f"❌ 로드 실패: {e}")
        st.stop()

    st.header("🚀 2. 분석 실행")
    if st.button("분석 시작", type="primary", use_container_width=True):
        log = []
        progress = st.progress(0)
        status = st.empty()

        status.text("▶ Step 1/6: 고립 노드 분리")
        degree = C.sum(axis=1)
        isolated = list(degree[degree == 0].index)
        active = list(degree[degree > 0].index)
        log.append(f"활성 {len(active)}개 / 고립 {len(isolated)}개")
        progress.progress(0.15)

        status.text("▶ Step 2/6: Louvain 해상도 스캔")
        C_act = C.loc[active, active]
        partition, best_res, scan_df = select_best_resolution(
            C_act, resolutions, independence_min, log
        )
        progress.progress(0.40)

        status.text("▶ Step 3/6: 크기 제약 적용")
        partition = enforce_max_size(C_act, partition, max_group_size, log)
        progress.progress(0.55)

        status.text("▶ Step 4/6: 소그룹 흡수")
        partition = absorb_small_groups(C_act, partition, min_group_size, log)
        progress.progress(0.75)

        status.text("▶ Step 5/6: 고립 노드 추가 + 재번호")
        partition = renumber_groups(partition, isolated)
        progress.progress(0.85)

        status.text("▶ Step 6/6: 결과 생성")
        group_df = group_stats(C, partition)
        total_int = group_df["내부연결"].sum()
        total_ext = group_df["외부연결"].sum() / 2
        overall_indep = total_int / (total_int + total_ext) * 100 if (total_int + total_ext) > 0 else 0
        log.append(f"✓ 최종: {len(group_df)}개 그룹, 전체 가중 독립도 {overall_indep:.1f}%")
        progress.progress(1.0)
        status.text("✅ 완료!")

        st.header("📊 3. 분석 결과")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("그룹 수", len(group_df))
        c2.metric("선택 해상도", best_res)
        c3.metric("전체 독립도", f"{overall_indep:.1f}%")
        c4.metric("고립 노드", len(isolated))

        tab1, tab2, tab3, tab4 = st.tabs(["📋 그룹 요약", "🔥 히트맵", "🔍 해상도 스캔", "📝 로그"])

        with tab1:
            st.dataframe(group_df, use_container_width=True, height=400)

        with tab2:
            with st.spinner("히트맵 생성 중..."):
                fig = make_heatmap(C, partition)
                st.pyplot(fig)

        with tab3:
            st.dataframe(scan_df, use_container_width=True)
            st.caption(f"✓ 최종 선택: resolution = {best_res}")

        with tab4:
            st.code("\n".join(log))

        st.header("💾 4. 결과 다운로드")
        excel_bytes = to_excel_bytes(group_df, C, partition)
        st.download_button(
            label="📥 엑셀 다운로드",
            data=excel_bytes,
            file_name="하우징_그룹핑_결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
else:
    st.info("👆 엑셀 파일을 업로드하세요")
    with st.expander("📖 사용법"):
        st.markdown("""
        1. 왼쪽 사이드바에서 파라미터 조정
        2. 엑셀 파일 업로드 (첫 시트에 매트릭스, 1행/1열=하우징명)
        3. [분석 시작] 버튼 클릭
        4. 결과 확인 및 다운로드
        """)
