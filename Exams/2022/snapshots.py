from fields import Fields

type1 = Fields(D=1, q=1, p=0.5, dt=0.01)
type1.run_show(20000, 100, save="D1_q1_p0.5.gif")

type1 = Fields(D=0.5, q=1, p=2.5, dt=0.01)
type1.run_show(20000, 100, save="D0.5_q1_p2.5.gif")