using LinearAlgebra
using Statistics

"""
    initialize_centroids_plusplus(data::Matrix{T}, k::Int) where {T<:AbstractFloat}

Initialize k centroids using k-means++.
Selects initial centroids that are spread out by choosing subsequent centroids
with probability proportional to squared distance from nearest existing centroid.
"""
function initialize_centroids_plusplus(data::Matrix{T}, k::Int) where {T<:AbstractFloat}
    n_samples, n_features = size(data)
    centroids = zeros(T, k, n_features)

    # Step 1: Choose first centroid uniformly at random
    first_idx = rand(1:n_samples)
    centroids[1, :] = data[first_idx, :]

    # Step 2: Choose remaining centroids using k-means++ strategy
    for i in 2:k
        # Compute distance from each point to nearest existing centroid
        min_distances = fill(Inf, n_samples)

        for j in 1:n_samples
            # Find minimum distance to any existing centroid
            for c in 1:(i-1)
                dist = sum((data[j, :] .- centroids[c, :]).^2)
                min_distances[j] = min(min_distances[j], dist)
            end
        end

        # Choose next centroid with probability proportional to squared distance
        # Points far from existing centroids are more likely to be selected
        probabilities = min_distances ./ sum(min_distances)
        cumsum_probs = cumsum(probabilities)
        r = rand()
        next_idx = findfirst(cumsum_probs .>= r)

        centroids[i, :] = data[next_idx, :]
    end

    return centroids
end

"""
    assign_to_nearest_centroid(data::Matrix{T}, centroids::Matrix{T}) where {T<:AbstractFloat}

Assign each data point to its nearest centroid using squared Euclidean distance.

Currently assigns points randomly. You need to:
1. Compute squared Euclidean distance from each point to each centroid
2. Assign each point to its nearest centroid
"""
function assign_to_nearest_centroid(data::Matrix{T}, centroids::Matrix{T}) where {T<:AbstractFloat}
    n_samples = size(data, 1)
    k = size(centroids, 1)
    labels = zeros(Int, n_samples)
  
    #for each data point
    for i in 1:n_samples
        min_distance = Inf
        nearest_centroid = 1
        #for each centroid
        for j in 1:k
          #get squared euclidean distance to each centroid
            distance = sum((data[i, :] .- centroids[j, :]).^2)

            #assign each point to nearest centroid
            if distance < min_distance
                min_distance = distance
                nearest_centroid = j
            end
        end
        labels[i] = nearest_centroid
    end
    return labels
end

"""
    update_centroids(data::Matrix{T}, labels::Vector{Int}, k::Int) where {T<:AbstractFloat}

Update each centroid to be the mean of all points assigned to that cluster.

Currently picks a random point from each cluster. You need to:
1. Find all points assigned to that cluster
2. Compute the mean of those points
3. Update the centroid to this mean

Hint: If a cluster is empty, leave its centroid unchanged.
"""
function update_centroids(data::Matrix{T}, labels::Vector{Int}, k::Int) where {T<:AbstractFloat}
    n_features = size(data, 2)
    centroids = zeros(T, k, n_features)

    for cluster in 1:k
        # Find points in this cluster
        cluster_mask = labels .== cluster
        cluster_points = data[cluster_mask, :]

        if size(cluster_points, 1) > 0
            # computes mean of all cluster points and updates centroid
            centroids[cluster, :] = mean(cluster_points, dims=1)
        end
    end

    return centroids
end

"""
    compute_wcss(data::Matrix{T}, labels::Vector{Int}, centroids::Matrix{T}) where {T<:AbstractFloat}

Compute within-cluster sum of squares (WCSS).
Sum of squared distances from each point to its assigned centroid.
Lower values indicate tighter clusters.
"""
function compute_wcss(data::Matrix{T}, labels::Vector{Int}, centroids::Matrix{T}) where {T<:AbstractFloat}
    wcss = zero(T)
    n_samples = size(data, 1)

    for i in 1:n_samples
        cluster = labels[i]
        # Squared distance from point to its assigned centroid
        dist_sq = sum((data[i, :] .- centroids[cluster, :]).^2)
        wcss += dist_sq
    end

    return wcss
end

"""
    reorder_labels_by_size(labels::Vector{Int}, centroids::Matrix{T}, k::Int) where {T<:AbstractFloat}

Reorder clusters by size (largest first) for consistent output.
"""
function reorder_labels_by_size(labels::Vector{Int}, centroids::Matrix{T}, k::Int) where {T<:AbstractFloat}
    # Count points in each cluster
    counts = [sum(labels .== i) for i in 1:k]

    # Get ordering by cluster size (largest to smallest)
    size_order = sortperm(counts, rev=true)

    # Create mapping from old labels to new labels
    label_map = zeros(Int, k)
    for (new_label, old_label) in enumerate(size_order)
        label_map[old_label] = new_label
    end

    # Apply mapping to labels
    new_labels = [label_map[label] for label in labels]

    # Reorder centroids
    new_centroids = centroids[size_order, :]

    return new_labels, new_centroids
end

"""
    kmeans(data::Matrix{T}, k::Int; max_iter=200, tol=1e-4, verbose=false) where {T<:AbstractFloat}

Perform k-means clustering.
Returns named tuple: (labels, centroids, wcss, iterations, converged)
"""
function kmeans(data::Matrix{T}, k::Int; max_iter=200, tol=1e-4, verbose=false) where {T<:AbstractFloat}
    n_samples, n_features = size(data)

    # Validate inputs
    if k > n_samples
        error("Number of clusters (k=$k) cannot exceed number of samples ($n_samples)")
    end
    if k < 1
        error("Number of clusters must be at least 1")
    end

    # Initialize centroids using k-means++
    centroids = initialize_centroids_plusplus(data, k)

    if verbose
        println("Starting k-means with k=$k")
        println("Max iterations: $max_iter, Tolerance: $tol")
    end

    # Initialize tracking variables
    converged = false
    labels = zeros(Int, n_samples)

    # Main iteration loop
    for iter in 1:max_iter
        # E-step: Assign points to nearest centroid
        labels = assign_to_nearest_centroid(data, centroids)

        # M-step: Update centroids
        old_centroids = copy(centroids)
        centroids = update_centroids(data, labels, k)

        # Check for convergence
        max_shift = maximum(norm(centroids[i, :] .- old_centroids[i, :]) for i in 1:k)

        if verbose && (iter % 10 == 0 || iter == 1)
            println("Iteration $iter: max centroid shift = $(round(max_shift, digits=6))")
        end

        if max_shift < tol
            if verbose
                println("Converged after $iter iterations!")
            end

            # Reorder clusters by size before returning
            labels, centroids = reorder_labels_by_size(labels, centroids, k)

            return (
                labels=labels,
                centroids=centroids,
                wcss=compute_wcss(data, labels, centroids),
                iterations=iter,
                converged=true
            )
        end
    end

    # If we get here, we hit max_iter without converging
    if verbose
        println("Reached max iterations without full convergence")
    end

    # Reorder clusters by size before returning
    labels, centroids = reorder_labels_by_size(labels, centroids, k)

    return (
        labels=labels,
        centroids=centroids,
        wcss=compute_wcss(data, labels, centroids),
        iterations=max_iter,
        converged=false
    )
end
