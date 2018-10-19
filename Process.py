# Processing the data from Yelp Data Challenge

# Author: Boyang Lu, Zhangyang Wei, Jie Zhou

import json
import pickle

# Read in the review file
f = open("../dataset_complete/review.json")
review_data = []
for line in f:
	review_data.append(json.loads(line))

# Read in the business information file
f = open("../dataset_complete/business.json")
business_data = []
for line in f:
	business_data.append(json.loads(line))

# Find the 10 businesses with the most reviews
selected_data = []
for i in range(10):
	max = 0
	selected = business_data[0]
	for term in business_data:
		if (term["review_count"] > max):
			max = term["review_count"]
			selected = term
	selected_data.append(selected)
	print(selected["business_id"] + "\n")
	selected["review_count"] = 0

# build a dict of (business_id, review_id)
business_review = dict()
# build a dict of (review_id, review_content)
id_content = dict()
for i in range(10):
	text = []
	for term in review_data:
		if (term["business_id"] == selected_data[i]["business_id"]):
			text.append(term["review_id"])
			id_content[term["review_id"]] = term
	business_review[selected_data[i]["business_id"]] = text

with open('business_review.data', 'wb') as output:
    pickle.dump(business_review, output, pickle.HIGHEST_PROTOCOL)

with open('id_content.data', 'wb') as output:
    pickle.dump(id_content, output, pickle.HIGHEST_PROTOCOL)


