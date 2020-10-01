# KMeans Clustering

#### Kmeans

Kmeans Clustering is an unsupervised clustering algorithm in Machine Learning, where a n-Dimensional data, is clustered by labelling the data with respect to the distance between the data-point  average centroid over a loop with repeated iterations till there is no further update in the centroid, due to its convergence.

---

#### Optimization Interpretation

The Kmeans algorithm can be expressed as an optimization problem, where our objective function tries to find cluster centers such that, the distance between data points and their closest centers is minimal as possible, when portioned into K number of cluster components.  
![{\rm{C = min }}\sum\limits_{i = 1}^n {\sum\limits_{j = 1}^k {{a_{ij}}{{\left\| {{x_i} - {\mu _j}} \right\|}_2}} } {\rm{ subject }}{{\rm{C}}_j}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Crm%7BC%20%3D%20min%20%7D%7D%5Csum%5Climits_%7Bi%20%3D%201%7D%5En%20%7B%5Csum%5Climits_%7Bj%20%3D%201%7D%5Ek%20%7B%7Ba_%7Bij%7D%7D%7B%7B%5Cleft%5C%7C%20%7B%7Bx_i%7D%20-%20%7B%5Cmu%20_j%7D%7D%20%5Cright%5C%7C%7D_2%7D%7D%20%7D%20%7B%5Crm%7B%20subject%20%7D%7D%7B%7B%5Crm%7BC%7D%7D_j%7D)  
![\begin{array}{l} {\rm{Where ,}}\\ {a_{ij}} \in \left\{ {0,1} \right\},{\rm{ defining whether data point }}{x_i}{\rm{ belongs to cluster }}{{\rm{C}}_j} \end{array}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%5Cbegin%7Barray%7D%7Bl%7D%0A%7B%5Crm%7BWhere%20%2C%7D%7D%5C%5C%0A%7Ba_%7Bij%7D%7D%20%5Cin%20%5Cleft%5C%7B%20%7B0%2C1%7D%20%5Cright%5C%7D%2C%7B%5Crm%7B%20defining%20whether%20data%20point%20%7D%7D%7Bx_i%7D%7B%5Crm%7B%20belongs%20to%20cluster%20%7D%7D%7B%7B%5Crm%7BC%7D%7D_j%7D%0A%5Cend%7Barray%7D)  
![{\mu _i}{\rm{ denotes the cluster center of }}{{\rm{C}}_j},{\rm{ for the corresponding Euclidean Distance }}{a_{ij}}{\left\| {{x_i} - {\mu _j}} \right\|_2}](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=%7B%5Cmu%20_i%7D%7B%5Crm%7B%20denotes%20the%20cluster%20center%20of%20%7D%7D%7B%7B%5Crm%7BC%7D%7D_j%7D%2C%7B%5Crm%7B%20for%20the%20corresponding%20Euclidean%20Distance%20%7D%7D%7Ba_%7Bij%7D%7D%7B%5Cleft%5C%7C%20%7B%7Bx_i%7D%20-%20%7B%5Cmu%20_j%7D%7D%20%5Cright%5C%7C_2%7D)  

-----
#### Eculidean Distance
Euclidean distance of a point in n-dimensional space is given by the Pythagorean formula as below :
![d({\rm{p, q) = }}d({\rm{q, p) = }}\sqrt {{{\left( {{{\rm{q}}_1} - {{\rm{p}}_1}} \right)}^2} + ...... + {{\left( {{{\rm{q}}_n} - {{\rm{p}}_n}} \right)}^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=d%28%7B%5Crm%7Bp%2C%20q%29%20%3D%20%7D%7Dd%28%7B%5Crm%7Bq%2C%20p%29%20%3D%20%7D%7D%5Csqrt%20%7B%7B%7B%5Cleft%28%20%7B%7B%7B%5Crm%7Bq%7D%7D_1%7D%20-%20%7B%7B%5Crm%7Bp%7D%7D_1%7D%7D%20%5Cright%29%7D%5E2%7D%20%2B%20......%20%2B%20%7B%7B%5Cleft%28%20%7B%7B%7B%5Crm%7Bq%7D%7D_n%7D%20-%20%7B%7B%5Crm%7Bp%7D%7D_n%7D%7D%20%5Cright%29%7D%5E2%7D%7D%20)  
Our data set belongs to 2-D space or R<sup>2</sup> space, hence the distance between two points is given as below :
![d({\rm{p, q) = }}d({\rm{q, p) = }}\sqrt {{{\left( {{{\rm{q}}_1} - {{\rm{p}}_1}} \right)}^2} + {{\left( {{{\rm{q}}_2} - {{\rm{p}}_2}} \right)}^2}} ](http://chart.apis.google.com/chart?cht=tx&chs=1x0&chf=bg,s,FFFFFF00&chco=000000&chl=d%28%7B%5Crm%7Bp%2C%20q%29%20%3D%20%7D%7Dd%28%7B%5Crm%7Bq%2C%20p%29%20%3D%20%7D%7D%5Csqrt%20%7B%7B%7B%5Cleft%28%20%7B%7B%7B%5Crm%7Bq%7D%7D_1%7D%20-%20%7B%7B%5Crm%7Bp%7D%7D_1%7D%7D%20%5Cright%29%7D%5E2%7D%20%2B%20%7B%7B%5Cleft%28%20%7B%7B%7B%5Crm%7Bq%7D%7D_2%7D%20-%20%7B%7B%5Crm%7Bp%7D%7D_2%7D%7D%20%5Cright%29%7D%5E2%7D%7D%20)
