import torch
import clip
import numpy as np

class ClipTransformer:

    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.torch_device)
        assert isinstance(self.preprocess, object)

    def _get_encoding(self, preprocessed_data, transform_type='image'):
        with torch.no_grad():
            if transform_type == 'image':
                features = self.model.encode_image(preprocessed_data)
            else:
                features = self.model.encode_text(preprocessed_data)
        features /= features.norm(dim=-1, keepdim=True)
        return features

    def _get_vecs_from_text(self, text_input):
        texts_preprocessed = torch.cat([clip.tokenize(c) for c in text_input]).to(self.torch_device)
        text_features = self._get_encoding(texts_preprocessed, transform_type='text')
        return text_features

    def _get_vecs_from_image(self, image_path):
        image_preprocessed = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.torch_device)
        image_features = self._get_encoding(image_preprocessed, transform_type='image')
        return image_features

    def fit_transform(self, inputs=[], transform_type='image'):

        results = []

        for input in tqdm(inputs):
            if transform_type == 'image':
                embedding = self._get_vecs_from_image(input)
                prefix = 'clip_image_'
            else:
                embedding = self._get_vecs_from_text(input)
                prefix = 'clip_text_'
            results.append((input, np.array(embedding)[0]))
        results = pd.DataFrame(results, columns=['input', 'vector'])
        vector_results = results['vector'].apply(pd.Series)
        self.results = pd.concat([results['input'], vector_results], axis=1).set_index('input')

        self.results.columns = [prefix + str(i) for i in self.results.columns]

        return self.results

class CLIPZeroShotClassifier:

    def __init__(self):
        self.model, self.torch_device, self.preprocess = self._initialise_clip()

    def _initialise_clip(self):
        """
        Initialises the CLIP model
        :return:model, torch, preprocess
        """
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=torch_device)
        assert isinstance(preprocess, object)
        return model, torch_device, preprocess

    def _transform_image_features(self, image_features: np.array):
        """
        Loads the image features as an array and converts them to a tensor
        :param image_features: Image CLIP vectors for classification
        :return: tensor_features: Image features as a tensor
        """
        tensor_features = torch.from_numpy(image_features).float()
        return tensor_features

    def _get_text_features(self, classes: str):
        """
        This tokenizes user text labels and then encodes them into CLIP vectors
        :param classes: classes for classification
        :return: text features
        """
        texts_tokenized = torch.cat([clip.tokenize(c) for c in classes]).to(self.torch_device)
        # Calculate features
        with torch.no_grad():
            text_features = self.model.encode_text(texts_tokenized)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _get_similarity(self, image_features: np.array, text_features: np.array):
        """
        Calculates the similarity between text and image vectors
        :param image_features: image vectors
        :param text_features: text vectors
        :return: similarity values and inices
        """
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(len(text_features))
        values = [i.item() for i in values]
        return values, indices

    def _get_ranked_labels(self, classes: str, indices):
        """
        This ranks the labels
        :param classes: classes for classification
        :param indices: indices of the results
        :return ranked_labels: the results ranked
        """
        ranked_labels = [classes[i] for i in indices]
        return ranked_labels

    def predict(self, image_features: np.array, classes: str):
        """
        This function uses all other functions to classify the image using the labels provided
        :param image_features: image vectors
        :param classes: classes for classification
        :return results_dict: the results
        """
        image_features = self._transform_image_features(image_features)
        text_features = self._get_text_features(classes)
        values, indices = self._get_similarity(image_features, text_features)
        ranked_labels = self._get_ranked_labels(classes, indices)
        results_dict = dict(zip(ranked_labels, values))
        results_dict = {k: round(v, 3) for k, v in results_dict.items()}
        return results_dict
