# Keyword Extraction

# Author: Boyang Lu, Zhangyang Wei, Jie Zhou

import pickle

# dictionary of (business id, list of review ids)
with open('business_review.data', 'rb') as input:
	business_review = pickle.load(input)

# dictionary of (review id, review contents)
with open('id_content.data', 'rb') as input:
	id_content = pickle.load(input)

# Example:
# Get the first review_id of the business '4JNXUYY8wbaaDmk3BPzlWw'
review_id = business_review['DkYS3arLOhA8si5uUEmHOw'][0]
# Get the content(text) of this review
text = id_content[review_id]['text']
# Get the stars of this review
stars = id_content[review_id]['stars']

print(text)
print(stars)

''' list of ten businesses with the most reviews
[
'4JNXUYY8wbaaDmk3BPzlWw',
'RESDUcs7fIiihp38-d6_6g',
'K7lWdNUhCbcnEvI0NhGewg',
'cYwJA2A6I12KNkm2rtXd5g',
'DkYS3arLOhA8si5uUEmHOw',
'f4x1YBxkLrZg652xt2KR5g',
'2weQS-RnoOBhb1KsHKyoSQ',
'KskYqH1Bi7Z_61pH6Om8pg',
'eoHdUeQDNgQ6WYEnP2aiRw',
'ujHiaprwCQ5ewziu0Vi9rw'
]
'''