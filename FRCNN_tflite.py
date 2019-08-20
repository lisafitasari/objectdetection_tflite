import kfp
from kfp import components
from kfp import dsl
from kfp import gcp
from kfp import compiler
import os

frcnn_aihub = components.load_component_from_url('https://storage.googleapis.com/cloud-aihub-service-prod/asset/public/products/40806618-18c4-4839-8684-e8c5202b8ba9/1/pipeline.tar.gz')
os.system('gsutil cp gs://kubeflow-pipelinev1/component/export_component_latest.yaml .')
export_tflite_op = components.load_component_from_file('export_component_latest.yaml')

@dsl.pipeline(
    name='FRCNN and Export to TFlite',
    description='FRCNN Object detection using AI HUB'
)

# Use a function to define the pipeline.
def frcnn_tflite(
    training-data-path='gs://images_pama/txt/class_ch2.txt',
    output-dir='gs://kubeflow-pipelinev1/frcnn_tflite_output',
    job-timeout-minutes='60',
    train-device='gpu',
    num-train-device='1',
    use-pretrained='True',
    use-low-precision='False',
    train-batch-size='2',
    num-examples-per-epoch='4',
    num-summary-per-epoch='1',
    num-epochs='1',
    learning-rate='0.08',
    first-lr-drop-epoch='8',
    second-lr-drop-epoch='11',
    image-size='640',
    num-classes='1',
    model_frcnn='gs:/kubeflow-pipelinev1/frcnn_tflite_output',
    project_id='satu-pama-ai-comex-and-tyre',
    output_tflite='gs://kubeflow-pipelinev1/tflite_output'
    ):

    # Use the component you loaded in the previous step to create a pipeline task.
    frcnn_aihub = frcnn_object_detection(
        training-data-path, 
        output-dir, 
        job-timeout-minutes,
        train-device,
        num-train-device,
        use-pretrained,
        use-low-precision,
        train-batch-size,
        num-examples-per-epoch,
        num-summary-per-epoch,
        num-epochs,
        learning-rate,
        first-lr-drop-epoch,
        second-lr-drop-epoch,
        image-size,
        num-classes)
    
    export_tflite = export_tflite_op(
        model_frcnn=model_frcnn,
        project_id=project_id,
        output_tflite=output_tflite
    )
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(kubeflow_training, __file__ + '.zip')