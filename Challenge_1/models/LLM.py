from transformers import T5Tokenizer, T5ForConditionalGeneration


class LLM:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    def generate_labels(self, training_dataset, test_dataset):
        # give some example using the training dataset
        examples = []
        for img, label in training_dataset:
            examples.append("image: "+str(img) + "has a cactus so the label is 1" if label == 1
                            else "image: "+str(img) + "doesn't have a cactus so the label is 0")
        # classify the new set of images
        return_values = []
        for batch in test_dataset:
            images_name, images, _ = batch
            for img_name, img in zip(images_name, images):
                inputs = self.tokenizer(
                    str(examples) + "\nclassify this image and if it has a cactus give me 1 otherwise 0" + str(img),
                    return_tensors="pt")
                outputs = self.model.generate(**inputs)
                output = self.tokenizer.decode(outputs[0])
                print(output)
                return_values.append((img_name, output))
        return return_values
