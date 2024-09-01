using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class SegmentationAnythingDemo : MonoBehaviour
{
    [SerializeField] private MeshRenderer _meshRenderer;
    [SerializeField] private Texture2D _texture;
    [SerializeField] private ModelAsset _encorder;
    [SerializeField] private ModelAsset _decorder;
    
    void Start()
    {
        var encordModel = ModelLoader.Load(_encorder);
        Tensor<float> inputTensor = TextureConverter.ToTensor(_texture);
        inputTensor.Reshape(new TensorShape(1, 3, 1024, 1024));
        
        Worker worker = new Worker(encordModel, BackendType.GPUPixel);
        
        worker.Schedule(inputTensor);
        
        Tensor<float> embedded = worker.PeekOutput(0) as Tensor<float>;
        Tensor<float> highResFeats0 = worker.PeekOutput(1) as Tensor<float>;
        Tensor<float> highResFeats1 = worker.PeekOutput(2) as Tensor<float>;
        
        Debug.Log($"embedded {embedded.shape}");
        Debug.Log($"highResFeats0 {highResFeats0.shape}");
        Debug.Log($"highResFeats1 {highResFeats1.shape}");

        Tensor<float> pointCoords = new Tensor<float>(new TensorShape(1, 1, 2), new float[] { 0.5f, 0.2f });
        Tensor<float> pointLabels = new Tensor<float>(new TensorShape(1, 1), new float[] { 1 });
        Tensor<float> maskInput = new Tensor<float>(new TensorShape(1, 1, 256, 256));
        Tensor<float> hasMask = new Tensor<float>(new TensorShape(1));
        Tensor<int> origImSize = new Tensor<int>(new TensorShape(2), new int[] { 1024, 1024 });

        var decordModel = ModelLoader.Load(_decorder);
        worker = new Worker(decordModel, BackendType.CPU);
        worker.Schedule(embedded, highResFeats1, highResFeats0, pointCoords, pointLabels, maskInput, hasMask, origImSize);
        var predicate = worker.PeekOutput(0);
        var mask = worker.PeekOutput(1);
        
        Debug.Log($"predicate {predicate.shape}");
        Debug.Log($"mask {mask.shape}");

        var maskTexture = TextureConverter.ToTexture(mask as Tensor<float>, 1024, 1024, 3, true);
        _meshRenderer.material.mainTexture = maskTexture;
    }
}
