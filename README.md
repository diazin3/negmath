# negmath
Negative Mathematics Playground

###### ------- WARNING - TAKE ALL THIS WITH A --H U G E-- GRAIN OF SALT -------#######
###### ------- DISCLAIMER: THIS WORK IS ALMOST ENTIRELY AI GENERATED THROUGH MY GUIDANCE, CODE AND THEORECTIC BACKGROUND INCLUDED ------######
###### ------- IM BUILDING THIS MORE OUT OF MERE CURIOSITY/FUN ------######
###### ------- https://substack.com/@negmath for more of this "work" ($8/mo :p)

# NegMath: Operator-Inversion Diagnostics for Complex Systems

**A lightweight, non-homomorphic probe that reveals structural stress in nonlinear PDEs and algebraic systems before they blow up.**

NegMath works by locally inverting selected operators (addition → subtraction, multiplication → division, etc.) to create "shadow" fields and equations. Where the shadow explodes, zeros out, or produces structured NaNs — that's where the original system ***potentially*** carries the most fragility.

### Why it matters
- Can be extremely cheap to compute depending on the terms that are being negmathed in the original equation.
- Apparently acts like a **structural MRI** for Navier-Stokes, turbulence, and other stiff nonlinear problems.
- Apparently gives early warning signals before your solver hits Inf/NaN or catastrophic amplitudes.
- Apparntly complements (and often precedes) spectral bottleneck or palinstrophy signals.

### Quick Demo (3D Incompressible Navier-Stokes, the very same scripts hosted in this repo)

```bash
# 1. Generate data (short violent run)
python 3D_NS_DATA_GENERATOR.py

# 2. Analyze and produce MRI plots
python DATA_ANALYZER.py
