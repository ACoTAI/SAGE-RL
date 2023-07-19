Robust Temporal Link Prediction in Dynamic Complex Networks via Stable Gated Networks Based on Reinforcement Learning
============================================
A **PyTorch** implementation of **SAGE-GCN**. 



### Requirements
The codebase is implemented in Python 3.7.1. package versions used for development are just below.
```
networkx          2.4
numpy             1.15.4
torch             1.7
tqdm              1.7
scipy             1.7.2
pandas            1.3.5
```
### Datasets
<p align="justify">
There are two dynamic email networks, called as ENRON and RADOSLAW. Each node stands for an employee in a mid-sized company and each edge appears to represent an email being sent from one person to another. ENRON records email interactions for nearly six months and RADOSLAW lasts for nearly nine months.<br>
http://networkrepository.com
</p>


Every .edges file has the following structure:

```javascript
from_node to_node  timestamp weight
    48 		13 	   926389620   1
    67      13     926418960   1
    67      13     926418960   1
    68      39     926418968   0
    ...

