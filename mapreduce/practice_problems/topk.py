"""

# Test case
documents = [
    ("doc1", "the cat sat on the mat"),
    ("doc2", "the dog ran in the park"),
    ("doc3", "cat and dog are pets")
]
k = 3
# Expected top 3: [("the", 4), ("cat", 2), ("dog", 2)]


"""

from typing import Tuple 


if __name__ == "__name__": 
    # Test case
    documents = [
        ("doc1", "the cat sat on the mat"),
        ("doc2", "the dog ran in the park"),
        ("doc3", "cat and dog are pets")
    ]
    k = 3
    # Expected top 3: [("the", 4), ("cat", 2), ("dog", 2)]
    topk_results = get_topk(documents)
    print(topk_results)