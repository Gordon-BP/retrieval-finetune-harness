## How to use

This UI can fine-tune two separate models that work together in tandem to retrieve relevant documents.

### Definitions

**Bi Encoder**- A transformer model that turns text into dense vectors (tensors).
**Corpus**- A collection of knowledge used to train and evaluate models
**Cosine Similarity**- A metric for calculating how close two vectors are by measuring the cosine of the angle between the two vectors.
**Cross Encoder**- A transformer model that takes two or more pieces of text and assigns a label. 
**Dot Product Similarity**- A metric for calculating how close two vectors are by multiplying the two vectors together and recording the product.
**Euclidean Distance**- A metric for calculating how close two vectors are by measuring the real distance between two vectors using the pythagorean theorem. 
**HNSW**- **H**eirarchal **N**eighbor **S**mall **W**orld is a method for quickly finding the cosine similarity between a query and a large corpus. It trades off a small bit of accuracy for speed.