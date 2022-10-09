from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, ClassificationMetrics, Artifact
from google_cloud_pipeline_components.v1.dataflow import DataflowPythonJobOp
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp
from kfp.v2 import compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from typing import NamedTuple
from datetime import datetime

PROJECT_ID = 'eternal-flux-364617'
REGION = 'us-central1'
ROOT_BUCKET = 'gs://bucket_cs_mlops'
BQ_DATASET = 'glass_classification'
BQ_TABLE = 'glass'
ENDPOINT = 'glassclassificaiton_endpoint'
MONITOR_EMAIL = 'pierluigidibari@gmail.com'

@component(
    packages_to_install=['google-cloud-bigquery','db-dtypes','pandas','sklearn'],
    base_image='python:3.7',
    output_component_file='getGlassDataFromBigQuery.yaml'
)
def getGlassDataFromBigQuery(
    projectid: str,
    region: str,
    bqdataset: str,
    bqtable: str,
    gcp_resources: str,
    training_set: Output[Dataset],
    test_set: Output[Dataset],
):
    
    from google.cloud import bigquery
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    client = bigquery.Client(
        project=projectid,
        location=region)
    
    query = "SELECT * FROM {}.{}.{}".format(projectid,bqdataset,bqtable)
    
    query_job = client.query(
        query,
        location=region)
    
    df = query_job.to_dataframe()
    
    train, test = train_test_split(
        df,
        random_state=0,
        stratify=df['Type'])
    
    train.to_csv(training_set.path + ".csv" , index=False)
    test.to_csv(test_set.path + ".csv" , index=False)

@component(
    packages_to_install=['pandas','sklearn'],
    base_image='python:3.7',
    output_component_file='trainGlassModel.yaml'
)
def trainGlassModel(
    training_set:  Input[Dataset],
    model: Output[Model], 
):
    
    import pandas as pd
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    training_set = pd.read_csv(training_set.path + '.csv')
    
    abc = AdaBoostClassifier(
        base_estimator=RandomForestClassifier(),
        random_state=0)
    
    abc.fit(training_set.drop(columns='Type'), training_set.Type)
    
    model.metadata["framework"] = 'ADA'
    
    file_name = model.path + '.pkl'
    with open(file_name, 'wb') as file:  
        pickle.dump(abc, file)

@component(
    packages_to_install=['pandas','sklearn'],
    base_image='python:3.7',
    output_component_file='evaluateGlassModel.yaml'
)
def evaluateGlassModel(
    test_set:  Input[Dataset],
    trained_model: Input[Model],
    metrics: Output[ClassificationMetrics],
) -> NamedTuple("output", [("deploy", str)]):

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import logging 
    import pickle
    import typing

    test = pd.read_csv(test_set.path + '.csv')
    abc = AdaBoostClassifier(
        base_estimator=RandomForestClassifier())
    file_name = trained_model.path + '.pkl'
    with open(file_name, 'rb') as file:  
        abc = pickle.load(file)
    
    x_test = test.drop(columns=['Type'])
    y_target=test.Type
    y_pred = abc.predict(x_test)
        
    metrics.log_confusion_matrix(
        ['1','2','3','4','5','6','7'],
        confusion_matrix(y_target, y_pred, labels=[1,2,3,4,5,6,7]).tolist())    
        
    accuracy = abc.score(x_test, y_target)
    trained_model.metadata["accuracy"] = float(accuracy)

    deploy = 'true' if accuracy > 0.5 else 'false'
    return (deploy,)

@component(
    packages_to_install=['google-cloud-aiplatform', 'scikit-learn==1.0.0',  'kfp'],
    base_image='python:3.7',
    output_component_file='deployGlassModel.yml'
)
def deployGlassModel(
    projectid: str,
    region: str,
    endpoint: str,
    model: Input[Model],
    serving_container_image_uri : str, 
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
) -> NamedTuple("output", [("endpoint", str)]):
    
    from google.cloud import aiplatform
    aiplatform.init(project=projectid, location=region)

    DISPLAY_NAME  = "glassclassification"
    MODEL_NAME = "glassclassification-ada"
    ENDPOINT_NAME = endpoint
    
    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(ENDPOINT_NAME),
        order_by='create_time desc',
        project=projectid, 
        location=region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]  # most recently created
        else:
            endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
        return endpoint
            
    endpoint = create_endpoint()   
                
    #Import a model programmatically
    model_upload = aiplatform.Model.upload(
        display_name = DISPLAY_NAME, 
        artifact_uri = model.uri.replace("/model", ""),
        serving_container_image_uri = serving_container_image_uri,
        serving_container_health_route= f"/v1/models/{MODEL_NAME}",
        serving_container_predict_route= f"/v1/models/{MODEL_NAME}:predict",
        serving_container_environment_variables={"MODEL_NAME": MODEL_NAME}       
    )
    
    model_deploy = model_upload.deploy(
        machine_type="n1-standard-4", 
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    # Save data to the output params
    vertex_model.uri = model_deploy.resource_name
    return (endpoint.name,)

@component(
    packages_to_install=['google-cloud-aiplatform', 'kfp'],
    base_image='python:3.7',
    output_component_file='monitorGlassModel.yml'
)
def monitorGlassModel(
    projectid: str,
    region: str,
    endpoint: str,
    monitoremail: str,
    training_set:  Input[Dataset]
):
    from google.cloud import aiplatform
    aiplatform.init(project=projectid, location=region)

    thresholds = {'RI':0.001,'Na':0.001,'Mg':0.001,'Al':0.001,'Si':0.001,'K':0.001,'Ca':0.001,'Ba':0.001,'Fe':0.001}
        
    skew_config = aiplatform.model_monitoring.SkewDetectionConfig(
        data_source= training_set.uri + '.csv',
        skew_thresholds=thresholds,
        attribute_skew_thresholds=thresholds,
        target_field='Type',
        data_format='csv'
    )

    drift_config = aiplatform.model_monitoring.DriftDetectionConfig(
        drift_thresholds=thresholds,
        attribute_drift_thresholds=thresholds,
    )

    objective_config = aiplatform.model_monitoring.ObjectiveConfig(
        skew_config,
        drift_config)

    # Create sampling configuration
    random_sampling = aiplatform.model_monitoring.RandomSampleConfig()

    # Create schedule configuration
    schedule_config = aiplatform.model_monitoring.ScheduleConfig(monitor_interval=1)

    # Create alerting configuration.
    # emails = ['pierluigidibari@gmail.com']
    alerting_config = aiplatform.model_monitoring.EmailAlertConfig(
        user_emails=[monitoremail], enable_logging=True
    )

    # Create the monitoring job.
    try:
        job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name='glassclassification_monitoring',
            logging_sampling_strategy=random_sampling,
            schedule_config=schedule_config,
            alert_config=alerting_config,
            objective_configs=objective_config,
            project=projectid,
            location=region,
            endpoint=endpoint)
    except:
        print('Monitoring job already present. Try delete it with !gcloud beta ai model-monitoring-jobs delete')

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = 'pipeline-glassclassification-job{}'.format(TIMESTAMP)

@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=ROOT_BUCKET + '/vertex_pipeline',
    # A name for the pipeline. Use to determine the pipeline Context.
    name="pipeline-glassclassification"    
)
def pipeline(
    projectid: str = PROJECT_ID,
    region: str = REGION, 
    rootbucket: str = ROOT_BUCKET,
    display_name: str = DISPLAY_NAME,
    bqdataset: str = BQ_DATASET,
    bqtable: str = BQ_TABLE,
    endpoint: str = ENDPOINT,
    monitoremail: str = MONITOR_EMAIL,
    dataflowtemp: str = ROOT_BUCKET + '/dataflow/temp',
    dataflowpy: str = ROOT_BUCKET + '/dataflow_pipeline/glassclassification_dataflow_pipeline.py',
    dataflowreq: str = ROOT_BUCKET + '/dataflow_pipeline/requirements.txt',
    args: list = [
        '--projectid', PROJECT_ID,
        '--region', REGION,
        '--rootbucket', ROOT_BUCKET,
        '--bqdataset', BQ_DATASET,
        '--bqtable', BQ_TABLE],
    api_endpoint: str = REGION + '-aiplatform.googleapis.com',
    serving_container_image_uri: str = 'europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest'
    ):
        
    dataflow_python_op = DataflowPythonJobOp(
        project=projectid,
        location=region,
        temp_location= dataflowtemp,
        python_module_path= dataflowpy,        
        requirements_file_path= dataflowreq,
        args = args)

    dataflow_wait_op = WaitGcpResourcesOp(
        gcp_resources=dataflow_python_op.outputs['gcp_resources'])
    
    data_op = getGlassDataFromBigQuery(
        projectid=projectid,
        region=region,
        bqdataset=bqdataset,
        bqtable=bqtable,
        gcp_resources=dataflow_wait_op.outputs['gcp_resources'])
    
    train_model_op = trainGlassModel(data_op.outputs["training_set"])
    
    model_evaluation_op = evaluateGlassModel(
        test_set=data_op.outputs["test_set"],
        trained_model=train_model_op.outputs["model"]
    )
    
    with dsl.Condition(
        model_evaluation_op.outputs["deploy"]=="true",
        name="deploy-glassclassification",
    ):
           
        deploy_model_op = deployGlassModel(
            projectid=projectid,
            region=region,
            endpoint=endpoint,
            model=train_model_op.outputs['model'],
            serving_container_image_uri = serving_container_image_uri)
        
        monitor_model_op = monitorGlassModel(
            projectid=projectid,
            region=region,
            monitoremail=monitoremail,
            endpoint=deploy_model_op.outputs['endpoint'],
            training_set=data_op.outputs['training_set'])

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='ml_glassclassification.json')

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="glassclassification-pipeline",
        template_path="ml_glassclassification.json",
        enable_caching=True, # True
        location=REGION)

    start_pipeline.run()