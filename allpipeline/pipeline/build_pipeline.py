from sklearn.pipeline import Pipeline
from allpipeline.pipeline.image_processors import image_processors
from allpipeline.pipeline.classifiers import classifiers

def build_pipeline(image_processor_labels, classifier_label) :
    '''
    |
    |   Create the pipeline which consists of 
    |   image processing step(s) and a classifier
    |
    '''

    estimators = []
    for label in image_processor_labels :
        estimators.append((label, image_processors[label]))
    estimators.append((classifier_label, classifiers[classifier_label]))

    return Pipeline(estimators)

