import numpy as np

PI_4 = np.pi / 4

def franka_analytic_ik(t_ee_0: np.array, q7: np.array, q_actual: np.array) -> np.array:
    """
        Ported from https://github.com/ICRA2026/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/panda_kinematics.py#L91

        Compute the inverse kinematics of the Franka Emika Panda robot.

        IK is adapted from the analytical solution found in:
        https://github.com/ffall007/franka_analytical_ik/blob/main/franka_ik_He.hpp

        Args:
        t_ee_0: The end-effector transformation matrix in base frame reference.
        q7: The joint 7 angle.
        q_actual: The actual current joint angles of the robot.
    """

    q = np.array([0.0] * 7, dtype=np.float32)

    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840

    # d7e is based on if hand or flange is used
    d7e = 0.2104
    # d7e = 0.107

    a4 = 0.0825
    a7 = 0.0880

    LL24 = 0.10666225
    LL46 = 0.15426225
    L24 = 0.326591870689
    L46 = 0.392762332715

    thetaH46 = 1.35916951803
    theta342 = 1.31542071191
    theta46H = 0.211626808766

    q_min = np.array(
      [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    )
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    q[6] = q7

    c1_a = np.cos(q_actual[0])
    s1_a = np.sin(q_actual[0])
    c2_a = np.cos(q_actual[1])
    s2_a = np.sin(q_actual[1])
    c3_a = np.cos(q_actual[2])
    s3_a = np.sin(q_actual[2])
    c4_a = np.cos(q_actual[3])
    s4_a = np.sin(q_actual[3])
    c5_a = np.cos(q_actual[4])
    s5_a = np.sin(q_actual[4])
    c6_a = np.cos(q_actual[5])
    s6_a = np.sin(q_actual[5])

    As_a = []

    As_a.append(
      np.array([
          [c1_a, -s1_a, 0.0, 0.0],
          [s1_a, c1_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, d1],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [c2_a, -s2_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [-s2_a, -c2_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [c3_a, -s3_a, 0.0, 0.0],
          [0.0, 0.0, -1.0, -d3],
          [s3_a, c3_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [c4_a, -s4_a, 0.0, a4],
          [0.0, 0.0, -1.0, 0.0],
          [s4_a, c4_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [1.0, 0.0, 0.0, -a4],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [c5_a, -s5_a, 0.0, 0.0],
          [0.0, 0.0, 1.0, d5],
          [-s5_a, -c5_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    As_a.append(
      np.array([
          [c6_a, -s6_a, 0.0, 0.0],
          [0.0, 0.0, -1.0, 0.0],
          [s6_a, c6_a, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
      ])
    )

    Ts_a = []
    Ts_a.append(As_a[0])
    for j in range(1, 7):
        Ts_a.append(np.matmul(Ts_a[j - 1], As_a[j]))

    # identify q6 case
    V62_a = Ts_a[1][:3, 3] - Ts_a[6][:3, 3]
    V6H_a = Ts_a[4][:3, 3] - Ts_a[6][:3, 3]
    Z6_a = Ts_a[6][:3, 2]
    is_case6_0 = np.sum(np.matmul(np.cross(V6H_a, V62_a), Z6_a)) <= 0

    # identify q1 case
    is_case1_1 = q_actual[1] < 0

    # IK: compute p_6
    R_EE = t_ee_0[:3, :3]
    z_EE = t_ee_0[:3, 2]
    p_EE = t_ee_0[:3, 3]
    p_7 = p_EE - (d7e * z_EE)

    x_EE_6 = np.array([np.cos(q7 - PI_4), -np.sin(q7 - PI_4), 0.0])
    x_6 = np.matmul(R_EE, x_EE_6)
    x_6 /= np.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    # IK: compute q4
    p_2 = np.array([0.0, 0.0, d1])
    V26 = p_6 - p_2

    LL26 = np.sum(V26 * V26)
    L26 = np.sqrt(LL26)

    theta246 = np.arccos((LL24 + LL46 - LL26) / 2.0 / L24 / L46)
    q[3] = theta246 + thetaH46 + theta342 - 2.0 * np.pi

    # IK: compute q6
    theta462 = np.arccos((LL26 + LL46 - LL24) / 2.0 / L26 / L46)
    theta26H = theta46H + theta462
    D26 = -L26 * np.cos(theta26H)

    Z_6 = np.cross(z_EE, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.column_stack(
      (x_6, Y_6 / np.linalg.norm(Y_6), Z_6 / np.linalg.norm(Z_6))
    )
    V_6_62 = np.matmul(R_6.T, -V26)

    Phi6 = np.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = np.arcsin(
      D26 / np.sqrt((V_6_62[0] * V_6_62[0]) + (V_6_62[1] * V_6_62[1]))
    )

    # q = np.where(
    #   is_case6_0, q.at[5].set(np.pi - Theta6 - Phi6), q.at[5].set(Theta6 - Phi6)
    # )
    if is_case6_0:
        q[5] = np.pi - Theta6 - Phi6
    else:
        q[5] = Theta6 - Phi6

    # q = jp.where(q[5] <= q_min[5], q.at[5].set(q[5] + 2.0 * jp.pi), q)
    if q[5] <= q_min[5]:
        q[5] = q[5] + 2.0 * np.pi

    # q = jp.where(q[5] >= q_max[5], q.at[5].set(q[5] - 2.0 * jp.pi), q)
    if q[5] >= q_max[5]:
        q[5] = q[5] - 2.0 * np.pi

    # IK: compute q1 & q2
    thetaP26 = 3.0 * np.pi / 2 - theta462 - theta246 - theta342
    thetaP = np.pi - thetaP26 - theta26H
    LP6 = L26 * np.sin(thetaP26) / np.sin(thetaP)

    z_6_5 = np.array([np.sin(q[5]), np.cos(q[5]), 0.0])
    z_5 = np.matmul(R_6, z_6_5)
    V2P = p_6 - LP6 * z_5 - p_2

    L2P = np.linalg.norm(V2P)

    V2P_L2P = np.abs(V2P[2] / L2P)
    greater_than_1 = np.where(V2P_L2P > 0.999, True, False)

    # q = jp.where(
    #   greater_than_1,
    #   q.at[0].set(q_actual[0]),
    #   q.at[0].set(jp.arctan2(V2P[1], V2P[0])),
    # )
    if greater_than_1:
        q[0] = q_actual[0]
    else:
        q[0] = np.arctan2(V2P[1], V2P[0])

    # q = jp.where(
    #   greater_than_1, q.at[1].set(0.0), q.at[1].set(jp.arccos(V2P[2] / L2P))
    # )
    if greater_than_1:
        q[1] = 0.0
    else:
        q[1] = np.arccos(V2P[2] / L2P)

    is_q_less_than_0 = np.where(q[0] < 0.0, True, False)
    # q = jp.where(
    #   is_case1_1,
    #   jp.where(
    #       is_q_less_than_0, q.at[0].set(q[0] + jp.pi), q.at[0].set(q[0] - jp.pi)
    #   ),
    #   q,
    # )
    if is_case1_1:
        if is_q_less_than_0:
            q[0] = q[0] + np.pi
        else:
            q[0] = q[0] - np.pi

    # q = jp.where(greater_than_1, q, jp.where(is_case1_1, q.at[1].set(-q[1]), q))
    if not greater_than_1:
        if is_case1_1:
            q[1] = -q[1]

    # IK: compute q3
    z_3 = V2P / L2P
    Y_3 = -np.cross(V26, V2P)
    y_3 = Y_3 / np.linalg.norm(Y_3)
    x_3 = np.cross(y_3, z_3)
    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    R_1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])

    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    R_1_2 = np.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
    R_2 = np.matmul(R_1, R_1_2)
    x_2_3 = np.matmul(R_2.T, x_3)
    # q = q.at[2].set(jp.arctan2(x_2_3[2], x_2_3[0]))
    q[2] = np.arctan2(x_2_3[2], x_2_3[0])

    # IK: compute q4
    VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5
    c6 = np.cos(q[5])
    s6 = np.sin(q[5])
    R_5_6 = np.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])
    R_5 = np.matmul(R_6, R_5_6.T)
    V_5_H4 = np.matmul(R_5.T, VH4)

    # q = q.at[4].set(-jp.arctan2(V_5_H4[1], V_5_H4[0]))
    q[4] = -np.arctan2(V_5_H4[1], V_5_H4[0])

    return q

