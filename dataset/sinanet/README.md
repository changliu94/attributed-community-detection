# Sinanet

Sinanet dataset is a microblog user relationship network extracted from sina-microblog website, http://www.weibo.com. We first selected 100 VIP sina-microblog users distributed in 10 major forums including finance and economics, literature and arts, fashion and vogue, current events and politics, sports, science and technology, entertainment, parenting and education, public welfare, etc. Starting from 100 VIP sina-microblog users, we extracted their followers/followees of these users and their published micro-blogs. Using depth first search strategy, we extracted three-layer of user relationships and obtained 8452 users, 147653 user relationships, and 5.5 million micro-blogs in total. We merged all microblogs that a user published to characterize the user's interests. After removing silent users which published less than 5000 words, we left with 3490 users and 30282 relationships. If we use words' frequency of the merged blogs of a user to describe the user's interest, the dimension of the feature space would be too high to be processed. We used users' topic distribution in the 10 forums, which was obtained by LDA topic model using the tool at http://gibbslda.sourceforge.net, to describe users' interests. Thus, besides the follower/followee relationships between pairs of users, we have 10 dimensional numerical attributes to describe the interests of each user.

edge.txt contains total 30282 edges representing the relationships between pairs of users.

content.txt contains the interest distriubtion of each user in 10 forums.

clusters.txt contains 10 clusters. Users in each cluser share the same class label, and belong to the same forum.

If you use the data set, please cite the paper, Node Attributes Enhanced Community Detection in Complex Networks, by Caiyan Jia et al., Scientific Report, 2017
