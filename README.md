#### vllm-test

Going to outline how to deploy a vLLM server for model fine-tuned with Unsloth with Kubernetes.  I chose to use Google's [G2 machines](https://cloud.google.com/blog/products/compute/introducing-g2-vms-with-nvidia-l4-gpus) which work exclusively with the NVIDIA L4 chips which are supposedly very good for inference workloads.  I simply experimented with [different sizes](https://cloud.google.com/compute/docs/gpus#l4-gpus) until I found one that fit my model. 


Important documentation: [vllm serve](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve)
^ This details all the arguments that you'll need to pass to `vllm serve`


Set some environment variables:
```
export CLUSTER_NAME="vllm-test"
export REGION="us-central1"
export VLLM_API_KEY="scr1b3pt"
```

First, we're going to create our cluster:
```
gcloud container clusters create $CLUSTER_NAME \
    --project=$PROJECT_ID \
    --region=$REGION \
    --workload-pool=$PROJECT_ID.svc.id.goog \
    --release-channel=rapid \
    --num-nodes=1
```

Next, we're going to create a node pool whose instances have GPU(s) attached to them.  Number and size of GPUs will depend on the size of the model that we're deploying of course.
```
gcloud container node-pools create gpupool \
    --accelerator type=nvidia-l4,count=2,gpu-driver-version=latest \
    --project=$PROJECT_ID \
    --location=$REGION \
    --node-locations=$REGION-a \
    --cluster=$CLUSTER_NAME \
    --machine-type=g2-standard-24 \
    --num-nodes=1
```

Now that we have a running cluster, we're going to configure our local kubeconfig (`~/.kube/config`) so that `kubectl` can communicate with our cluster.
```
gcloud container clusters get-credentials $CLUSTER_NAME --location=$REGION
```

Finally, we're going to create a secret containing our HuggingFace API token.  vLLM will need this API token to pull models from the HuggingFace model hub.
```
kubectl create secret generic hf-secret \
    --from-literal=hf_api_token=$HF_TOKEN \
    --dry-run=client -o yaml | kubectl apply -f -
```
```
kubectl create secret generic vllm-secret \
    --from-literal=vllm_api_key=$VLLM_API_KEY \
    --dry-run=client -o yaml | kubectl apply -f -
```

Now, we're going to do our K8s deployment:
```
kubectl apply -f vllm_k8s.yaml
kubectl wait --for=condition=Available --timeout=600s deployment/vllm-deployment
```

Review the logs and make sure we're good!
```
kubectl logs -f -l app=vllm-server
```

Set up port forwarding:
```
kubectl port-forward service/llm-service 8000:8000
```

Now, in a seperate tab, let's test it out!

```
export TEST_PROMPT="### Conversation:
Doctor: What brings you back into the clinic today, miss? 
Patient: I came in for a refill of my blood pressure medicine. 
Doctor: It looks like Doctor Kumar followed up with you last time regarding your hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis and kidney stones.  Have you noticed any changes or do you have any concerns regarding these issues?  
Patient: No. 
Doctor: Have you had any fever or chills, cough, congestion, nausea, vomiting, chest pain, chest pressure?
Patient: No.  
Doctor: Great. Also, for our records, how old are you and what race do you identify yourself as?
Patient: I am seventy six years old and identify as a white female.

### Header:
General History

### Summary:"

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer scr1b3pt" \
    -X POST \
    -d @- <<EOF
{
    "model": "donaldrauscher/medical-scribe-vllm",
    "prompt": "${TEST_PROMPT}",
    "temperature": 0.0,
    "max_tokens": 500
}
EOF
```