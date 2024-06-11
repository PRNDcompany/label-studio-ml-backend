import numpy as np
import cv2

from label_studio_converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase


class CvtPolygon2BrushBackend(LabelStudioMLBase):

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
       
        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            print('[]')
            return []
        
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        # collect context information
        predictions = []
        points = []
        selected_label = None
        for ctx in context['result']:
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'polygonlabels':
                if ctx['value']['closed']:
                    points = ctx['value']['points']
                    if len(points) == 0:
                        continue

                    predictor_results = self._convert_polygon_to_brush(
                        points=points,
                        width=image_width,
                        height=image_height
                    )

                    predictions.append(
                        self.get_results(
                            masks=predictor_results['masks'],
                            probs=predictor_results['probs'],
                            width=image_width,
                            height=image_height,
                            from_name=from_name,
                            to_name=to_name,
                            label=selected_label
                        )[0]
                    )
        print(len(predictions))
        return predictions[-1:]
    
    def _convert_polygon_to_brush(self, points, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = points * np.array([width, height]) / np.array([100, 100])
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
        mask = mask.astype(bool)
        return {
            'masks': [mask],
            'probs': [1.0]
        }

    def get_results(self, masks, probs, width, height, from_name, to_name, label):
        results = []
        total_prob = 0
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })
        return [{
            'result': results,
            'model_version': self.get('model_version'),
            'score': total_prob / max(len(results), 1)
        }]
