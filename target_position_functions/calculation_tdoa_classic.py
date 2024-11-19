import os
import matplotlib.pyplot as plt
from utils.time_utils import calculate_time_difference
from utils.plots_utils import plot_hyperbola_contour, hyperbola_contour_all_options

import numpy as np
from numpy.linalg import norm, inv
from numpy import abs, sin, cos, isreal, isfinite, roots, zeros, concatenate, pi, arctan2, cross, dot, real


def TDOA_3points(rov_lst, t_lst, velocity, num_file, is_plot=False, is_extreme=False, depth=None, is_3d=False):
    # The function moves and rotates the points so that the second point is at
    # the beginning of the axes and the third point is at the X-axis (on the
    # right side of the X-axis).
    rov_lst, R, dif = Triangular_rotation(rov_lst)
    t_b, t_a, t_c = t_lst
    p_b, p_a, p_c = rov_lst

    dt_ab, dt_ac = calculate_time_difference(t_lst) * np.array([1, -1])

    # The point that is not on the X-axis will always have a positive Y value
    idx_flip = False
    if p_b[1] < 0:
        idx_flip = True
        p_b[1] = -1 * p_b[1]

    if (depth is None) or (not is_3d):
        depth = 0

    # Calculation of TDOA hyperbola parameters
    RD_ab, c_ab, NRD_ab, A_abx, B_aby = calc_parm(p_a, p_b, dt_ab, velocity)
    RD_ac, c_ac, NRD_ac, A_acx, B_acy = calc_parm(p_a, p_c, dt_ac, velocity)

    # if is_extreme:
    #     RD_ab, c_ab, NRD_ab, A_abx, B_aby = calc_parm_extreme(p_a, p_b, t_a, t_b, velocity)
    #     RD_ac, c_ac, NRD_ac, A_acx, B_acy = calc_parm_extreme(p_a, p_c, t_a, t_c, velocity)

    # Calculate the lengths of the sides of the triangle
    ab, ac, bc = norm(p_a - p_b), norm(p_a - p_c), norm(p_b - p_c)

    # plots the dots
    fig1, ax1 = None, None
    if is_plot:
        fig1, ax1 = plt.subplots()
        name_list = ['R_a (p2)', 'R_b (Ref)', 'R_c (p3)']
        p_list = np.array([p_a, p_b, p_c]) - np.array([c_ac, 0])
        for i in range(3):
            ax1.scatter(p_list[i, 0], p_list[i, 1])
            ax1.text(p_list[i, 0] + .4, p_list[i, 1] + .4, name_list[i], fontsize=9)
        ax1.grid()

    # Calculate the angle between the points
    alpha = 0 - np.arccos((0 - bc ** 2 + ab ** 2 + ac ** 2) / (2 * ab * ac))

    # Move the points so that the line between the two points on
    # the X-axis is in the middle of the beginning of the axes
    P_a = p_a - np.array([c_ac, 0])
    P_b = p_b - np.array([c_ac, 0])
    P_c = p_c - np.array([c_ac, 0])

    # Hyperbolic plotting
    if is_plot:
        x, y = np.linspace(-50, 50, 400), np.linspace(-50, 50, 400)
        x, y = np.meshgrid(x, y)

        # plots all possible hyperbolas
        hyperbola_contour_all_options(ax1, x, y, RD_ac, RD_ab, alpha, A_acx, B_acy, A_abx, B_aby, c_ab, c_ac)

        # plots only relevant hyperbolas [ ) or ( ]
        plot_hyperbola_contour(ax1, x, y, P_b, P_a, RD_ab, 'pink')
        plot_hyperbola_contour(ax1, x, y, P_c, P_a, RD_ac, 'pink')

    # If the object is not at the same distance between any two points of
    # the triangle - a standard TDOA calculation is examined.
    res = None
    if (RD_ab != 0) & (RD_ac != 0):
        x = intersection_hyperbolas(A_acx, B_acy, A_abx, B_aby, c_ac, c_ab, 0 - alpha, depth)
        z_ac = 1 + (depth ** 2) / B_acy
        y = (B_acy * ((x ** 2 / A_acx ** 2) - z_ac)) ** 0.5  # y = (B_acy * ((x ** 2 / A_acx ** 2) - 1)) ** 0.5
        z = y * 0 + depth
        res = concatenate((np.array([x, y, z]).T, np.array([x, -y, z]).T), axis=0)
    # If the object is at the same distance between two points of the triangle,
    # we will calculate the point of intersection between the hyperbola and a linear line.
    elif (RD_ab != 0) & (RD_ac == 0):
        y = intersection_hyperbola_liner_ac(A_abx, B_aby, c_ac, c_ab, 0 - alpha, depth)
        x = y * 0
        z = y * 0 + depth
        res = np.array([x, y, z]).T
    # Similar to the previous case
    elif (RD_ab == 0) & (RD_ac != 0):
        res = intersection_hyperbola_liner_ab(A_acx, B_acy, P_a, P_b, depth)
    # If the object is at the same distance between any two points of the triangle,
    # we will calculate the circum-center of the triangle
    elif (RD_ab == 0) & (RD_ac == 0):
        res = circumcenter(P_a, P_b, P_c, depth)

    # We may receive more solutions, we will test our
    # solutions according to the relevant functions
    check1 = norm(res - np.array([P_a[0], P_a[1], 0]), axis=1) - \
             norm(res - np.array([P_b[0], P_b[1], 0]), axis=1) - RD_ab
    check2 = norm(res - np.array([P_a[0], P_a[1], 0]), axis=1) - \
             norm(res - np.array([P_c[0], P_c[1], 0]), axis=1) - RD_ac
    res = res[abs(check1 + check2) < 0.1, :]

    # Plot all possible solutions for object location
    if is_plot:
        ax1.scatter(res[:, 0], res[:, 1])
        ax1.set_xlim([-30, 30])
        ax1.set_ylim([-10, 30])
        fig1.show()
        fig_path = os.path.join("../plots", "full_plt" + str(num_file) + '.jpeg')
        fig1.savefig(fig_path)

    # In case no solution is received, we will return an empty list
    if res.size == 0:
        return np.array([])

    # There will be cases where we may get up to 2 possible solutions.
    # In this case we have no way of determining which solution is better so we will use what we prefer
    if res.shape[0] > 1:
        print("Warning: You have received more than one possible locations for the target")
        # "the task will select the first location")

    # We return the solution to the original coordinate base, and add a depth dimension to the solution
    res = Triangular_rotation_rev(res[:, :2] + np.array([c_ac, 0]), R, dif, idx_flip)  # temporary 2D without depth
    return np.append(res, zeros((res.shape[0], 1)) + depth, axis=1)


# If the object is at the same distance between any two points of the triangle,
# we will calculate the circum-center of the triangle
def circumcenter(p_a, p_b, p_c, depth=0):
    # TODO: check if it correct for 3d case
    ax, ay = p_a
    bx, by = p_b
    cx, cy = p_c

    A = (ax ** 2 + ay ** 2)
    B = (bx ** 2 + by ** 2)
    C = (cx ** 2 + cy ** 2)
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    ux = (A * (by - cy) + B * (cy - ay) + C * (ay - by)) / D
    uy = (A * (cx - bx) + B * (ax - cx) + C * (bx - ax)) / D

    return np.array([[ux, uy, depth]])


# If the object is at the same distance between two points of the triangle,
# we will calculate the point of intersection between the hyperbola ac and a linear line ab (y=0).
def intersection_hyperbola_liner_ab(a, b, p_a, p_b, depth=0):
    z1 = 1 + (depth ** 2) / b

    mid_p = np.array([(p_b[0] + p_a[0]) / 2, (p_b[1] + p_a[1]) / 2])
    m = - (p_b[0] - p_a[0]) / (p_b[1] - p_a[1])
    k = mid_p[1] - m * mid_p[0]

    A = (a ** 2) * (m ** 2) - b
    B = 2 * (a ** 2) * m * k
    C = (a ** 2) * (k ** 2) + (a ** 2) * b * z1

    if all(isreal([A, B, C])) & all(isfinite([A, B, C])):
        x = roots([A, B, C])
        x = real(x[x.imag == 0])
        y = m * x + k
        z = y * 0 + depth
        res = concatenate((np.array([x, y, z]).T, np.array([x, -y, z]).T), axis=0)
    else:
        res = np.array([])

    return res


# If the object is at the same distance between two points of the triangle,
# we will calculate the point of intersection between the hyperbola ab and a linear line ac.
def intersection_hyperbola_liner_ac(f, d, h1, h2, alpha, depth=0):
    zz = 1 + (depth ** 2) / d

    alpha = 0 - alpha
    sin_a = sin(alpha)
    cos_a = cos(alpha)

    z1 = h1 * cos_a - h2
    z2 = h1 * sin_a

    a = d * (sin_a ** 2) - (f ** 2) * (cos_a ** 2)
    b = 0 - 2 * d * z1 * sin_a - 2 * (f ** 2) * z2 * cos_a
    c = d * (z1 ** 2) - (f ** 2) * (z2 ** 2) - (f ** 2) * d * zz

    if all(isreal([a, b, c])) & all(isfinite([a, b, c])):
        y = roots([a, b, c])
        y = real(y[y.imag == 0])
    else:
        y = np.array([])

    return y


# If the object is not at the same distance between any two points of
# the triangle - a standard TDOA calculation is examined.
def intersection_hyperbolas(a, b, f, d, h1, h2, alpha, depth):
    alpha = 0 - alpha
    sin_a = sin(alpha)
    cos_a = cos(alpha)

    z1 = 1 + (depth ** 2) / b
    z2 = 1 + (depth ** 2) / d

    al = d * (sin_a ** 2) - (f ** 2) * (cos_a ** 2)
    bl = 0 - 2 * (f ** 2) * sin_a * cos_a - 2 * d * sin_a * cos_a
    cl = 2 * d * h2 * sin_a
    dl = d * (cos_a ** 2) - (f ** 2) * (sin_a ** 2)
    el = 0 - 2 * d * h2 * cos_a
    fl = d * (h2 ** 2) - (f ** 2) * d * z2

    at = b / (a ** 2)
    gl = bl * h1 + cl

    K = at * (bl ** 2)
    L = 2 * at * bl * gl
    M = at * (gl ** 2) - b * (bl ** 2) * z1
    N = 0 - 2 * b * bl * gl * z1
    P = 0 - b * (gl ** 2) * z1

    q = (al * b) / (a ** 2)
    t = fl - al * b * z1

    ql = q + dl
    sl = 2 * dl * h1 + el
    tl = dl * (h1 ** 2) + el * h1 + t

    Kl = ql ** 2
    Ll = 2 * sl * ql
    Ml = 2 * ql * tl + sl ** 2
    Nl = 2 * sl * tl
    Pl = tl ** 2

    a4 = K - Kl
    a3 = L - Ll
    a2 = M - Ml
    a1 = N - Nl
    a0 = P - Pl

    if all(isreal([a0, a1, a2, a3, a4])) & all(isfinite([a0, a1, a2, a3, a4])):
        x = roots([a4, a3, a2, a1, a0])
        x = real(x[x.imag == 0])
    else:
        x = np.array([])

    return x


# The function moves and rotates the points so that the second point is at
# the beginning of the axes and the third point is at the X-axis (on the
# right side of the X-axis).
def Triangular_rotation(rov_lst):
    if abs(rov_lst[0, 1] - rov_lst[1, 1]) <= 1e-7:
        rov_lst[0, :] = rov_lst[0, :] + 1e-07

    # p1 = rov_lst[0]
    p2 = rov_lst[1]
    p3 = rov_lst[2]

    dif = p2
    check_minus = 2 * (0 - 1) * (p3[1] > 0) + 1

    # p1 = p1 - dif
    p2 = p2 - dif
    p3 = p3 - dif

    v_1 = concatenate((p3 - p2, zeros(1)), axis=0)
    v_2 = concatenate((np.array([check_minus, 0]) - p2, zeros(1)), axis=0)

    alpha, alpha1, alpha2 = 0, 0, 0
    rotation_option = 0
    if (p3[0] <= 0) & (p3[1] <= 0):
        alpha1 = arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        alpha2 = - pi - arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        rotation_option = 1
    elif (p3[0] >= 0) & (p3[1] <= 0):
        alpha1 = arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        alpha2 = pi - arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        rotation_option = 2
    elif (p3[0] <= 0) & (p3[1] >= 0):
        alpha1 = 0 - pi + arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        alpha2 = - arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        rotation_option = 3
    elif (p3[0] >= 0) & (p3[1] >= 0):
        alpha1 = pi + arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        alpha2 = - arctan2(norm(cross(v_1, v_2)), dot(v_1, v_2))
        rotation_option = 4

    R1 = np.array([[cos(alpha1), 0 - sin(alpha1)], [sin(alpha1), cos(alpha1)]])
    R2 = np.array([[cos(alpha2), 0 - sin(alpha2)], [sin(alpha2), cos(alpha2)]])

    if abs((R1 @ p3)[1]) < 0.1:
        alpha = alpha1
    elif abs((R2 @ p3)[1]) < 0.1:
        alpha = alpha2
    else:
        # print(rov_lst[2])
        print('Error - Triangular_rotation didnt work in section ' + str(rotation_option))

    R = np.array([[cos(alpha), 0 - sin(alpha)], [sin(alpha), cos(alpha)]])
    rov_lst = rov_lst - dif

    for i in range(rov_lst.shape[0]):
        rov_lst[i] = R @ rov_lst[i]

    return rov_lst, R, dif


def Triangular_rotation_rev(res, r_matrix, dif, idx_flip):
    if idx_flip:
        res[:, 1] = -1 * res[:, 1]

    for i in range(res.shape[0]):
        res[i] = inv(r_matrix) @ res[i] + dif

    return res


# def calc_parm_old(p1, p2, t1, t2, velocity):
#     RD = velocity * (t1 - t2)
#     c = norm(p1 - p2) / 2
#     NRD = RD / (2 * c)
#     Ax = abs(RD) / 2
#     By = c ** 2 - Ax ** 2
#
#     return RD, c, NRD, Ax, By


def calc_parm(p1, p2, dt, velocity):
    RD = velocity * dt
    c = norm(p1 - p2) / 2
    NRD = RD / (2 * c)
    Ax = abs(RD) / 2
    By = c ** 2 - Ax ** 2

    return RD, c, NRD, Ax, By


def calc_parm_extreme(p1, p2, dt_12, velocity, threshold=0.98):
    rd, c, nrd, ax, by = calc_parm(p1, p2, dt_12, velocity)

    if nrd >= threshold:  # extreme
        new_rd = 2 * c
        new_nrd = 1
    elif nrd <= 0 - threshold:
        new_rd = 0 - 2 * c
        new_nrd = -1
    else:
        new_rd = rd
        new_nrd = nrd

    ax = abs(new_rd) / 2
    by = c ** 2 - ax ** 2

    return new_rd, c, new_nrd, ax, by


def intersection_between_hyperbola_to_line(rov_lst, t_lst, velocity):
    t_a, t_b, t_c = t_lst
    p_a, p_b, p_c = rov_lst
    dt_ab, dt_bc = calculate_time_difference(t_lst)

    RD_ab, c_ab, NRD_ab, A_abx, B_aby = calc_parm_extreme(p_a, p_b, dt_ab, velocity)
    RD_bc, c_bc, NRD_bc, A_bcx, B_bcy = calc_parm_extreme(p_b, p_c, dt_bc, velocity)

    if abs(NRD_ab) >= 1 and abs(NRD_bc) >= 1:
        m1 = (p_b[1] - p_a[1]) / (p_b[0] - p_a[0])
        m2 = (p_c[1] - p_b[1]) / (p_c[0] - p_b[0])

        n1 = p_b[1] - m1 * p_b[0]
        n2 = p_b[1] - m2 * p_b[0]

        x = (n2 - n1) / (m1 - m2)
        y = m1 * x + n1

        res = np.array([[x, y, 0]])

    elif abs(NRD_ab) >= 1 or abs(NRD_bc) >= 1:
        if abs(NRD_ab) >= 1:
            a = (p_b[1] - p_a[1]) / (p_b[0] - p_a[0])
            m = p_b[1] - a * p_b[0]
            b, c, d, e = p_b[0], p_b[1], p_c[0], p_c[1]
            R = RD_bc
        else:  # abs(NRD_bc) >= 1:
            a = (p_c[1] - p_b[1]) / (p_c[0] - p_b[0])
            m = p_b[1] - a * p_b[0]
            b, c, d, e = p_a[0], p_a[1], p_b[0], p_b[1]
            R = RD_ab

        A = 1 + (a ** 2)
        B = 2 * a * (m - c) - 2 * b
        C = (b ** 2) + (m - c) ** 2
        D = 2 * a * (m - e) - 2 * d
        E = (d ** 2) + (m - e) ** 2

        a2 = (B - D) ** 2 - 4 * (R ** 2) * A
        a1 = 2 * (B - D) * (R ** 2 + C - E) - 4 * (R ** 2) * B
        a0 = (R ** 2 + C - E) ** 2 - 4 * (R ** 2) * C
        x = roots([a2, a1, a0])
        y = a * x + m
        z = y * 0

        res = np.array([x, y, z]).T
    else:
        res = np.array([])

    idx_list = []
    if res.size > 0:
        for val in res:
            check_ab = (norm(val[0:2] - p_a) - norm(val[0:2] - p_b) >= 0) == (NRD_ab >= 0)
            check_bc = (norm(val[0:2] - p_b) - norm(val[0:2] - p_c) >= 0) == (NRD_bc >= 0)
            if check_ab and check_bc:
                idx_list.append(True)
            else:
                idx_list.append(False)

    return res[idx_list]
