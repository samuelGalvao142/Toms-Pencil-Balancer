import numpy as np

#"""Joint angles θ₁, θ₄ in radians, each in [0, 2π)."""

SRx = 198.95 #mm
SRy = 163.297 #mm
SLx = 174.061 #mm
SLy = 196.793 #mm

la = 175.838 #mm
lb = 175.05 #mm

## NEW Inverse Kinematics - 6/20/2026

def circle_intersections(x0, y0, r0, x1, y1, r1):
    # Distance between centers
    dx = x1 - x0
    dy = y1 - y0
    d = np.sqrt(dx*dx + dy*dy)

    # No solution cases
    if d > r0 + r1:
        return []  # circles too far apart

    if d < abs(r0 - r1):
        return []  # one circle inside the other

    if d == 0 and r0 == r1:
        return []  # infinite solutions

    # Distance from center 0 to midpoint line
    a = (r0**2 - r1**2 + d**2) / (2*d)

    # Height from midpoint to intersection
    h = np.sqrt(r0**2 - a**2)

    # Midpoint between intersections
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    # Intersection points
    xs1 = xm + h * (-dy) / d
    ys1 = ym + h * ( dx) / d

    xs2 = xm - h * (-dy) / d
    ys2 = ym - h * ( dx) / d

    return [(xs1, ys1), (xs2, ys2)]

def ik_solve(xd, yd):
    EL = circle_intersections(SLx, SLy, la, xd, yd, lb)
    ER = circle_intersections(SRx, SRy, la, xd, yd, lb)

    # skip invalid points
    if not EL or not ER:
        t1 = 94
        t4 = 97

    if EL and ER:
        #branch selection
        if EL[0][0] < EL[1][0]:
            ELx, ELy = EL[0]
        else:
            ELx, ELy = EL[1]

        if ER[0][0] > ER[1][0]:
            ERx, ERy = ER[0]
        else:
            ERx, ERy = ER[1]

        dxL = ELx - SLx
        dyL = ELy - SLy

        dxR = ERx - SRx
        dyR = ERy - SRy

        # global angles
        thetaL_global = np.degrees(np.arctan2(dyL, dxL))
        thetaR_global = np.degrees(np.arctan2(dyR, dxR))

        # LEFT: from +y axis, CCW+
        t1 = (thetaL_global - 90 + 360) % 360

        # RIGHT: from -x axis, CCW+
        t4 = (thetaR_global - 180 + 360) % 360

    return t1, t4