---
title: Computer Engineering Robotics 2025-2026
layout: home
---

1. [Github Pages Personal Portfolio](https://github.com/topics/jekyll-theme) [jekyll, markdown/yml]
2. [Jupyter Notebook Running in Python Environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) [pip/venv]
3. Linux (optional) [linux2windows or dual boot]
4. [Exporatory Data Analysis](https://datascienceguide.github.io/exploratory-data-analysis) [Matplotlib, Seaborn] 
5. [Introduction to Statistical Learning](https://www.statlearning.com/resources-python)
5a. Regression [Scikit learn, statsmodels]
5b. Classfication
5c. Clustering
7. Association [https://www.youtube.com/watch?v=YgnpqrgKTbE&list=PLUpgd_KWKlSBuI6-a-bSBd6NLewjlFAUc&index=6]
8. Search (TBD)
9. Reinforcement Learning (Programming an AI to play  a video game)
10.a. [Street Fighter](https://www.youtube.com/watch?v=rzbFhu6So5U)
Competitions:
10.b [Xfuzzycomp](https://xfuzzycomp.github.io/XFC/) [Python]
10.c. [battle code](https://battlecode.org/)  [Java]
10.d [starcraft ai](https://www.cs.mun.ca/~dchurchill/starcraftaicomp/)  [Java]

## Units and Equations Reference

### Force
- Newtons (N)  
- Pounds-force (lbf)

### Mass
- Pounds-mass (lbm)  
- Kilograms (kg)

### Time
- Seconds (s)  
- Minutes (min)  
- Hours (h)

### Distance
- Meters (m)

### Velocity:
$v = \frac{m}{s}$  

### Acceleration:
$a = \frac{m}{s^2}$  

### Energy
- Joules (J)  
- Calories (cal)

### Power:
$P = \frac{J}{s} = V \cdot A$  

### Pressure
- Pascals (Pa)  
- Pounds per square inch (psi)  
- Bar  
- Millimeters of mercury (mmHg)

### Torque
$$ \tau = N \cdot m $$
or in Imperial units:  
$$ \tau = ft \cdot lbf $$

### Force of Mr. Andrade on the Moon 

**Question:**  
What is the force of Mr. Andrade standing on the surface of the Moon in **lbf** if he weighs **190 lb** on Earth?

---
## **GUESS Method**

**Given:**  
$W_{\rm Earth} = 190~\rm{lb} = 190 \cdot 4.44822~\rm{N} \approx 844.2~\rm{N}$  
$g_{\rm Earth} = 9.81~\rm{m/s^2}$  
$g_{\rm Moon} = 1.625~\rm{m/s^2}$

**Unknown:**  
$F_{\rm Moon} = \rm{lbf}~?$  

**Equation:**  
$W = m \cdot g$  
$m = \frac{W_{\rm Earth}}{g_{\rm Earth}}$  
$F_{\rm Moon} = m \cdot g_{\rm Moon}$

**Substitute:**  
$m = \frac{844.2}{9.81} \approx 86.1~\rm{kg}$  
$F_{\rm Moon} = 86.1 \cdot 1.625 \approx 140~\rm{N}$

**Solve:**  
$F_{\rm Moon} \approx 140~\rm{N}$

$F_{\rm Moon} = 140~\rm{N} \cdot \frac{1~\rm{lbf}}{4.44822~\rm{N}} \approx 31.4~\rm{lbf}$


**∴**The force on Mr. Andrade on the Moon is approximately $\mathbf{31.4~lbf}$.

----

[^1]: [It can take up to 10 minutes for changes to your site to publish after you push the changes to GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll#creating-your-site).

[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
[GitHub Pages]: https://docs.github.com/en/pages
[README]: https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md
[Jekyll]: https://jekyllrb.com
[GitHub Pages / Actions workflow]: https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/
[use this template]: https://github.com/just-the-docs/just-the-docs-template/generate
