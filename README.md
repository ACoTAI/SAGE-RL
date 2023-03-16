SAGE-GCN
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
There are two dynamic email networks, called as ENRON and RADOSLAW, in which the nodes and edges represent the users and the e-mails.
FB-FORUM denotes a forum network of the same online community, where edges are the activities participated by two individuals.
    SFHH and INVS are the human contact network, which consists of persons and the real-world contact between two people.<br>
http://networkrepository.com
</p>


Every .edges file has the following structure:

```javascript
from_node to_node  timestamp weight
48 		  13 	   	 926389620   1
67        13     926418960   1
67        13     926418960   1
...
```
