Detects (1) all changes in a functional time series data (2) then filters out changes which are relevant to the research problem .


In (1) A binary segmentation is used. This algorithm recursively looks for a change point, splits the data at the change point and then looks for change points in each segment. The algorithm only stops when the statistical function lies below a certain threshold value. This threshold value can be fixed by the user. Alternatively, an automated procedure may be used for estimating the threshold.

In (2) the methods developed in the paper (https://arxiv.org/abs/2312.11108) are provided. This method uses the change locations detected in (1) and filters out a subset of changes which are relevant to the research question under study.
