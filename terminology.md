#### Terminology
The following terms will come up repeatedly in our discussion of effective machine learning:

**Instance:** The thing about which you want to make a prediction. For example, the instance
might be a web page that you want to classify as either "about cats" or "not about cats".

**Label:** An answer for a prediction task ­­ either the answer produced by a machine learning
system, or the right answer supplied in training data. For example, the label for a web page
might be "about cats".

**Feature:** A property of an instance used in a prediction task. For example, a web page might
have a feature "contains the word 'cat'".

**Feature Column:<sup>1</sup>** A set of related features, such as the set of all possible countries in which 1 user might live. An example may have one or more features present in a feature column. A feature column is referred to as a “namespace” in the VW system (at Yahoo/Microsoft), or a field.

**Example:** An instance (with its features) and a label.

**Model:** A statistical representation of a prediction task. You train a model on examples then use
the model to make predictions.

**Metric:** A number that you care about. May or may not be directly optimized.

**Objective:** A metric that your algorithm is trying to optimize.

**Pipeline:** The infrastructure surrounding a machine learning algorithm. Includes gathering the data from the front end, putting it into training data files, training one or more models, and exporting the models to production.

1 - Google specific terminology
