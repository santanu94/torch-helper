def accuracy(output_label_similarity_list):
    output_label_similarity_tensor = output_label_similarity_list[0]
    for batch_tensor in output_label_similarity_list[1:]:
        output_label_similarity_tensor = torch.cat((output_label_similarity_tensor, batch_tensor))
    return torch.tensor(torch.sum(output_label_similarity_tensor).item() / len(output_label_similarity_tensor)).item()
