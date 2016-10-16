In this problem set, I explored unsupervised learning with K-means and Hierarchical clustering.<br />
Looking at the USDA nutrient database, one would find various types of foods with their corresponding nutritional contents: http://ndb.nal.usda.gov/ndb/search/list. In this problem, we will experiment with 4 food groups: Cereal-Grain-Pasta, Finfish-Shellfish, Vegetables, Fats-Oils.<br /><br />
The data contains detailed categorizations of each food item. For example, there are 9 categories of Kale. They range from raw, frozen and unprepared, cooked and boiled, cooked and drained without salt, etc. In addition, common knowledge suggests that major food groups can be further categorized. Vegetables can be leaves/stems, roots, or buds. Based on their nutritional contents, one might expect to see these items clustered in hierarchies, from the major food groups, sub-groups, and finally to variants of the same food items.<br /><br />
The following data files were provided with the assignment:
  1. dataDescriptions.txt – gives the names of all the attributes (nutrients).
  2. dataCereal-grains-pasta.txt, dataFinfish-shellfish.txt, dataVegetables.txt, dataFats-oils.txt – the data files, one for each food group. <br /><br />
**NOTE**: The attributes in the data files are delimited by the caret (^) character.
