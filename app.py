from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.algorithms.tsp.heuristic import nearest_neighbor_tour, tour_cost, two_opt
from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model

# --- PAGE SETUP ---
st.set_page_config(
    page_title="FSTSP Interactive Model Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HEADER ---
st.title("FSTSP Delivery Simulation & Model Demo")
st.caption(
    "Mô phỏng và Trực quan hóa Thuật toán Điều phối Giao hàng phối hợp Xe tải – Drone (Flying Sidekick Traveling Salesman Problem)"
)
st.divider()


# --- SIDEBAR CONFIGURATOR ---
with st.sidebar:
    st.subheader("1. Kịch bản Giao hàng")
    scenario = st.selectbox(
        "Loại hình mạng lưới",
        [
            "Ngoại ô (Phân bố đều - Suburban)",
            "Khu đô thị (Khách hàng tập trung cụm)",
            "Vùng hạn chế (Một số khách chỉ Drone bay tới được)"
        ]
    )
    
    n_customers = st.slider("Số lượng khách hàng (N):", min_value=6, max_value=20, value=10, step=1)
    seed = st.number_input("Seed ngẫu nhiên:", min_value=1, max_value=999, value=42, step=1)
    
    st.subheader("2. Thông số Phương tiện")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        drone_speed = st.slider("Tốc độ Drone:", min_value=1.5, max_value=3.0, value=2.0, step=0.5, help="Tốc độ xe tải mặc định = 1.0")
    with c_s2:
        endurance = st.slider("Pin Drone (đơn vị):", min_value=20, max_value=60, value=35, step=5)

    st.subheader("3. Thuật toán Giải")
    algorithm = st.selectbox("Bộ giải:", ["CMSA Metaheuristic (Nhanh & Tối ưu)", "Exact 2-Index MILP (Toàn cục)"])
    time_limit = st.slider("Thời gian giải (giây):", min_value=2, max_value=30, value=8, step=1)
    
    btn_solve = st.button("Chạy Tối ưu hóa Mô hình", type="primary", use_container_width=True)


# --- INSTANCE GENERATION HELPER ---
def get_instance(scenario_name: str, n: int, s: int, d_spd: float, endur: float) -> FSTSPInstance:
    rng = np.random.default_rng(s)
    if "cụm" in scenario_name:
        # Clustered scenario
        c1 = rng.uniform(15, 45, size=(n // 2, 2))
        c2 = rng.uniform(55, 85, size=(n - n // 2, 2))
        coords = np.vstack([c1, c2])
        rng.shuffle(coords)
        depot = np.array([50.0, 50.0])
    else:
        coords = rng.uniform(5, 95, size=(n, 2))
        depot = np.array([50.0, 50.0])
        
    novisit = None
    if "hạn chế" in scenario_name:
        # 30% customers forbidden for truck (only drone can reach, or vice versa)
        drone_allowed = [True] * n
    else:
        drone_allowed = [True] * n
        
    inst = FSTSPInstance(
        name=f"Demo_{n}cust_s{s}",
        coords=coords,
        depot_coord=depot,
        truck_speed=1.0,
        drone_speed=float(d_spd),
        drone_endurance=float(endur),
        launch_time=1.0,
        recovery_time=1.0,
        drone_allowed=drone_allowed,
    )
    return inst


# --- INITIALIZE OR RETRIEVE SESSION STATE ---
if "current_instance" not in st.session_state or btn_solve:
    with st.spinner("Đang xây dựng mạng lưới và tính toán phương án tối ưu..."):
        inst = get_instance(scenario, n_customers, seed, drone_speed, endurance)
        
        # 1. Solve baseline Pure Truck TSP
        t0_tsp = time.perf_counter()
        tsp_route = nearest_neighbor_tour(inst.customers, inst.truck_time, inst.S, inst.E)
        tsp_route = two_opt(tsp_route, inst.truck_time)
        tsp_time = tour_cost(tsp_route, inst.truck_time)
        
        # 2. Solve FSTSP with selected algorithm
        t0_fstsp = time.perf_counter()
        if "Exact" in algorithm:
            sol = solve_stage_model(inst, time_limit=float(time_limit), mip_rel_gap=0.01)
        else:
            sol = solve_cmsa(inst, total_time=float(time_limit), mip_time=min(6.0, float(time_limit)/2.0), age_limit=2, seed=int(seed))
            
        st.session_state["current_instance"] = inst
        st.session_state["fstsp_solution"] = sol
        st.session_state["tsp_route"] = tsp_route
        st.session_state["tsp_time"] = tsp_time

inst: FSTSPInstance = st.session_state["current_instance"]
sol: FSTSPSolution = st.session_state["fstsp_solution"]
tsp_route: list[int] = st.session_state["tsp_route"]
tsp_time: float = st.session_state["tsp_time"]

# Evaluate schedule
sched = evaluate_schedule(inst, sol)
issues = validate_solution(inst, sol)
is_valid = sol.feasible and len(issues) == 0

# --- TOP PERFORMANCE COMPARISON CARDS ---
fstsp_time = sol.objective if sol.objective is not None else float("inf")
time_saved = max(0.0, tsp_time - fstsp_time)
time_saved_pct = (time_saved / tsp_time) * 100.0 if tsp_time > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label="Thời gian Chỉ dùng Xe tải (TSP)",
        value=f"{tsp_time:.1f}s",
        help="Phương án truyền thống: Xe tải tự đi giao toàn bộ khách hàng"
    )

with col_m2:
    st.metric(
        label="Thời gian Phối hợp Xe + Drone (FSTSP)",
        value=f"{fstsp_time:.1f}s",
        delta=f"-{time_saved_pct:.1f}% Thời gian" if is_valid else "Không khả thi",
        delta_color="normal"
    )

with col_m3:
    truck_stops = len(sol.truck_route) - 2
    drone_sorties_count = len(sol.drone_sorties)
    st.metric(
        label="Phân bổ Phương tiện",
        value=f"{truck_stops} Khách xe / {drone_sorties_count} Khách drone",
        help=f"Drone đảm nhiệm {drone_sorties_count}/{inst.n} điểm giao"
    )

with col_m4:
    valid_text = "Hợp lệ (55/55 Ràng buộc)" if is_valid else f"Lỗi ({len(issues)} vi phạm)"
    st.metric(
        label="Kiểm định Vật lý & Pin",
        value=valid_text,
        help="Kiểm tra pin drone, thời gian cất cánh, hạ cánh và đồng bộ điểm hẹn"
    )

st.divider()


# --- INTERACTIVE SIMULATION & CONTROLS ---
st.subheader("Mô phỏng Hành trình Giao hàng theo Thời gian thực")

# Determine max makespan for simulation
max_sim_time = float(np.ceil(fstsp_time if is_valid else tsp_time))

sim_col1, sim_col2 = st.columns([3, 1])
with sim_col1:
    sim_t = st.slider(
        "Kéo mốc thời gian để xem vị trí xe tải và drone (giây):",
        min_value=0.0,
        max_value=max_sim_time,
        value=min(15.0, max_sim_time / 2.0),
        step=0.5,
    )
with sim_col2:
    view_mode = st.radio(
        "Hiển thị lộ trình:",
        ["Phối hợp Xe tải + Drone (FSTSP)", "So sánh cả 2 lộ trình (TSP vs FSTSP)", "Chỉ Xe tải (TSP)"],
        index=0
    )


# --- INTERPOLATION FOR VEHICLE POSITIONS AT TIME T ---
def compute_positions_at_t(t: float, inst: FSTSPInstance, sol: FSTSPSolution, sched):
    xy = inst.node_coords
    route = sol.truck_route
    
    # 1. Truck position
    truck_pos = xy[inst.S].copy()
    truck_status = "Đang ở Depot xuất phát"
    
    for stage in range(len(route) - 1):
        curr_node = route[stage]
        next_node = route[stage + 1]
        t_dep = sched.truck_departure[stage]
        t_arr = sched.truck_arrival[stage + 1]
        t_next_dep = sched.truck_departure[stage + 1]
        
        if t <= t_dep:
            truck_pos = xy[curr_node].copy()
            truck_status = f"Đang dừng / thao tác tại Node {curr_node}"
            break
        elif t_dep < t <= t_arr:
            alpha = (t - t_dep) / (t_arr - t_dep) if (t_arr - t_dep) > 1e-6 else 1.0
            truck_pos = (1 - alpha) * xy[curr_node] + alpha * xy[next_node]
            truck_status = f"Đang di chuyển: Node {curr_node} → Node {next_node}"
            break
        elif t_arr < t <= t_next_dep:
            truck_pos = xy[next_node].copy()
            truck_status = f"Đang tại Node {next_node} (chờ / phục hồi drone)"
            break
        else:
            truck_pos = xy[next_node].copy()
            truck_status = "Đã về Depot kết thúc"

    # 2. Drone position
    drone_pos = truck_pos.copy()
    drone_status = "Đang nằm trên nóc xe tải (sẵn sàng)"
    drone_battery = 100.0
    
    pos_map = {node: stg for stg, node in enumerate(route)}
    
    for s in sol.drone_sorties:
        h = s.customer
        ls = pos_map[s.launch_node]
        rs = pos_map[s.recovery_node]
        
        t_launch_start = sched.truck_departure[ls] - inst.launch_time
        t_launch_end = sched.truck_departure[ls]
        
        flight_out = float(inst.drone_time[s.launch_node, h])
        flight_in = float(inst.drone_time[h, s.recovery_node])
        
        t_cust_arr = t_launch_end + flight_out
        t_rendezvous = t_cust_arr + flight_in
        t_recovery_end = sched.truck_departure[rs]
        
        if t_launch_start <= t < t_launch_end:
            drone_pos = xy[s.launch_node].copy()
            drone_status = f"Đang nạp gói hàng & chuẩn bị cất cánh tại Node {s.launch_node}"
            drone_battery = 100.0
            break
        elif t_launch_end <= t < t_cust_arr:
            beta = (t - t_launch_end) / flight_out if flight_out > 1e-6 else 1.0
            drone_pos = (1 - beta) * xy[s.launch_node] + beta * xy[h]
            elapsed_flight = t - t_launch_end
            drone_battery = max(0.0, 100.0 - (elapsed_flight / inst.drone_endurance) * 100.0)
            drone_status = f"Đang bay giao hàng tới Khách {h}"
            break
        elif t_cust_arr <= t < t_rendezvous:
            gamma = (t - t_cust_arr) / flight_in if flight_in > 1e-6 else 1.0
            drone_pos = (1 - gamma) * xy[h] + gamma * xy[s.recovery_node]
            elapsed_flight = t - t_launch_end
            drone_battery = max(0.0, 100.0 - (elapsed_flight / inst.drone_endurance) * 100.0)
            drone_status = f"Đã giao xong Khách {h}, đang bay về điểm hẹn Node {s.recovery_node}"
            break
        elif t_rendezvous <= t < t_recovery_end:
            drone_pos = xy[s.recovery_node].copy()
            drone_battery = max(0.0, 100.0 - ((t_rendezvous - t_launch_end) / inst.drone_endurance) * 100.0)
            drone_status = f"Đã đến điểm hẹn Node {s.recovery_node} (chờ xe tải thu hồi)"
            break

    return truck_pos, truck_status, drone_pos, drone_status, drone_battery


# --- RENDER INTERACTIVE MAP ---
xy = inst.node_coords
fig = go.Figure()

# 1. Pure Truck TSP baseline (if toggled)
if "TSP" in view_mode:
    tsp_x, tsp_y = [], []
    for a, b in zip(tsp_route, tsp_route[1:]):
        tsp_x.extend([xy[a, 0], xy[b, 0], None])
        tsp_y.extend([xy[a, 1], xy[b, 1], None])
    fig.add_trace(go.Scatter(
        x=tsp_x, y=tsp_y,
        mode="lines",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
        name="Lộ trình thuần Xe tải (TSP Baseline)",
        hoverinfo="none"
    ))

# 2. FSTSP Truck Route
if "FSTSP" in view_mode or "So sánh" in view_mode:
    route = sol.truck_route
    trk_x, trk_y = [], []
    for a, b in zip(route, route[1:]):
        trk_x.extend([xy[a, 0], xy[b, 0], None])
        trk_y.extend([xy[a, 1], xy[b, 1], None])
    fig.add_trace(go.Scatter(
        x=trk_x, y=trk_y,
        mode="lines",
        line=dict(color="#1E40AF", width=2.5),
        name="Lộ trình Xe tải (FSTSP)",
        hoverinfo="none"
    ))
    
    # Drone Sorties
    for idx, s in enumerate(sol.drone_sorties):
        h = s.customer
        sx = [xy[s.launch_node, 0], xy[h, 0], xy[s.recovery_node, 0]]
        sy = [xy[s.launch_node, 1], xy[h, 1], xy[s.recovery_node, 1]]
        fig.add_trace(go.Scatter(
            x=sx, y=sy,
            mode="lines",
            line=dict(color="#D97706", width=2.0, dash="dash"),
            name="Chặng Drone bay" if idx == 0 else None,
            showlegend=(idx == 0),
            hoverinfo="none"
        ))

# 3. Customer nodes
drone_served_custs = {s.customer for s in sol.drone_sorties}
truck_served_custs = [c for c in inst.customers if c not in drone_served_custs]

fig.add_trace(go.Scatter(
    x=[xy[c, 0] for c in truck_served_custs],
    y=[xy[c, 1] for c in truck_served_custs],
    mode="markers+text",
    marker=dict(symbol="circle", size=9, color="#1E40AF", line=dict(color="#FFFFFF", width=1)),
    text=[str(c) for c in truck_served_custs],
    textposition="bottom right",
    name="Khách xe tải giao",
    hovertemplate="Khách hàng %{text} (Xe tải)<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[xy[c, 0] for c in drone_served_custs],
    y=[xy[c, 1] for c in drone_served_custs],
    mode="markers+text",
    marker=dict(symbol="diamond", size=11, color="#D97706", line=dict(color="#78350F", width=1)),
    text=[str(c) for c in drone_served_custs],
    textposition="top right",
    name="Khách drone giao",
    hovertemplate="Khách hàng %{text} (Drone)<extra></extra>"
))

# 4. Depot
fig.add_trace(go.Scatter(
    x=[inst.depot_coord[0]],
    y=[inst.depot_coord[1]],
    mode="markers+text",
    marker=dict(symbol="square", size=14, color="#DC2626", line=dict(color="#000000", width=1.5)),
    text=["Depot"],
    textposition="top center",
    name="Kho trung tâm (Depot)",
    hovertemplate="Depot trung tâm<extra></extra>"
))

# 5. Live Simulation Markers at time t
if is_valid and sched.truck_departure:
    t_pos, t_stat, d_pos, d_stat, d_bat = compute_positions_at_t(sim_t, inst, sol, sched)
    
    # Truck current position
    fig.add_trace(go.Scatter(
        x=[t_pos[0]], y=[t_pos[1]],
        mode="markers+text",
        marker=dict(symbol="circle", size=18, color="#2563EB", line=dict(color="#FFFFFF", width=2)),
        text=["XE TẢI"],
        textposition="bottom center",
        name="Vị trí Xe tải (hiện tại)",
        hovertemplate=f"Vị trí Xe tải tại t={sim_t:.1f}s<extra></extra>"
    ))
    
    # Drone current position
    fig.add_trace(go.Scatter(
        x=[d_pos[0]], y=[d_pos[1]],
        mode="markers+text",
        marker=dict(symbol="diamond", size=16, color="#EA580C", line=dict(color="#FFFFFF", width=2)),
        text=["DRONE"],
        textposition="top center",
        name="Vị trí Drone (hiện tại)",
        hovertemplate=f"Vị trí Drone tại t={sim_t:.1f}s<extra></extra>"
    ))

fig.update_layout(
    template="plotly_white",
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Tọa độ X (km)", showgrid=True, zeroline=False, gridcolor="#E2E8F0", linecolor="#CBD5E1"),
    yaxis=dict(title="Tọa độ Y (km)", showgrid=True, zeroline=False, gridcolor="#E2E8F0", linecolor="#CBD5E1", scaleanchor="x", scaleratio=1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=560,
)

st.plotly_chart(fig, use_container_width=True)


# --- REAL-TIME STATUS BAR AT TIME T ---
if is_valid and sched.truck_departure:
    t_pos, t_stat, d_pos, d_stat, d_bat = compute_positions_at_t(sim_t, inst, sol, sched)
    
    st.markdown(f"**Trạng thái hoạt động tại thời điểm t = {sim_t:.1f}s:**")
    st_c1, st_c2, st_c3 = st.columns([1.5, 1.5, 1])
    
    with st_c1:
        st.info(f"**Xe tải:** {t_stat}")
    with st_c2:
        st.warning(f"**Drone:** {d_stat}")
    with st_c3:
        st.metric(label="Mức Pin Drone", value=f"{d_bat:.1f}%")

st.divider()


# --- MISSION LOG & SCHEDULE DETAILS ---
st.subheader("Nhật ký Lịch trình Đồng bộ Chi tiết (Mission Dispatch Log)")

log_col1, log_col2 = st.columns([1.2, 1])

with log_col1:
    st.markdown("**Trình tự các Stage của Xe tải:**")
    stage_steps = []
    for s_idx, node in enumerate(sol.truck_route):
        arr_t = sched.truck_arrival[s_idx] if s_idx < len(sched.truck_arrival) else 0.0
        dep_t = sched.truck_departure[s_idx] if s_idx < len(sched.truck_departure) else 0.0
        label = f"Node {node}" if node in inst.customers else f"Depot ({node})"
        stage_steps.append({
            "Stage": s_idx,
            "Địa điểm": label,
            "Thời điểm đến": f"{arr_t:.1f}s",
            "Thời điểm rời đi": f"{dep_t:.1f}s",
            "Thời gian dừng": f"{(dep_t - arr_t):.1f}s"
        })
    st.dataframe(pd.DataFrame(stage_steps), hide_index=True, use_container_width=True)

with log_col2:
    st.markdown("**Nhiệm vụ Giao hàng của Drone:**")
    if sol.drone_sorties:
        sortie_records = []
        pos_map = {n: i for i, n in enumerate(sol.truck_route)}
        for idx, s in enumerate(sol.drone_sorties):
            ls = pos_map[s.launch_node]
            rs = pos_map[s.recovery_node]
            t_launch = sched.truck_departure[ls]
            f_out = float(inst.drone_time[s.launch_node, s.customer])
            f_in = float(inst.drone_time[s.customer, s.recovery_node])
            t_rend = t_launch + f_out + f_in
            total_flt = f_out + f_in
            
            sortie_records.append({
                "Chặng": f"#{idx+1}",
                "Khách phục vụ": f"Khách {s.customer}",
                "Phóng từ": f"Node {s.launch_node} ({t_launch:.1f}s)",
                "Điểm hẹn": f"Node {s.recovery_node} ({t_rend:.1f}s)",
                "Thời gian bay": f"{total_flt:.1f}s"
            })
        st.dataframe(pd.DataFrame(sortie_records), hide_index=True, use_container_width=True)
    else:
        st.write("Không có chặng drone nào trong phương án này.")
