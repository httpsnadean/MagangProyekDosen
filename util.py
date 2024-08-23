def get_detections(net, blob):
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks

def load_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = f.read().strip().split("\n")
    return class_names
