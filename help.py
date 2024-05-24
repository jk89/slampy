from slampy.visual_model.train import VocTrain
from slampy.visual_model.train import VocTrain
from slampy.visual_model.train_results_converter import VocComputeModelGenerator

trainer = VocTrain()
trained_results = trainer.load_train_results_from_file("voctree_train_results_k_12_d_350000_m_hamming_t_2024_02_04_21_20_14.pkl")

# keep
def sum_level_data_counts(node):
    if node is None:
        return 0

    count = node.levelData.shape[0]# if hasattr(node, 'levelData') else 0

    if node.children:
        for child in node.children:
            count += sum_level_data_counts(child)

    return count

def count_word_layer_children(node):
    if node is None:
        return 0

    count = 1 if node.wordLayer else 0

    if node.children:
        for child in node.children:
            count += count_word_layer_children(child)

    return count

# keep
def count_all_children(node):
    if node is None:
        return 0

    count = len(node.children)

    for child in node.children:
        count += count_all_children(child)

    return count

def count_deepest_children(node):
    if node is None or len(node.children) == 0:
        return 0

    max_depth = 0
    for child in node.children:
        max_depth = max(max_depth, count_deepest_children(child))

    if max_depth == 0:
        return 1
    else:
        return max_depth



training_results_converter = VocComputeModelGenerator(trained_results)

out_thing = training_results_converter.get_handle_model()

print(count_word_layer_children(trained_results))
print(sum_level_data_counts(trained_results))
print("count_all_children out_thing which is compute model as a tree", count_all_children(out_thing))
print("count_deepest_children", count_deepest_children(out_thing))
def find_word_layer_no_children(node, indexes=[]):
    """
    Find the paths of nodes where word_layer_bool is False and there are no children.
    """
    if node is None:
        return

    # Check if the node's word_layer_bool is False and there are no children
    if not node.word_layer_bool and not node.children:
        print("Path:", indexes)  # Print the path to this node

    # Recursively search for children
    for i, child in enumerate(node.children):
        find_word_layer_no_children(child, indexes + [i])  # Append the current index to the path

print("check_word_layer_no_children(root_node)", find_word_layer_no_children(out_thing))


arahads = training_results_converter.pack_compute_model(out_thing)

print("training_results_converter.pack_compute_model(out_thing)", arahads)