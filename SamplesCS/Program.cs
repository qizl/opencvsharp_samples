﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCvSharp;
using SampleBase;

namespace SamplesCS
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Demo();
        }

        static void Demo()
        {
            ISample sample =
            new SimilarCompare();
            //new ArucoSample();
            //new BgSubtractorMOG();
            //new BinarizerSample();
            //new BRISKSample();
            //new CaffeSample();
            //new ClaheSample();
            //new ConnectedComponentsSample();
            //new DFT();
            //new FaceDetection();
            //new FaceDetectionDNN();
            //new FASTSample();
            //new FlannSample(); 
            //new FREAKSample();
            //new HandPose();
            //new HistSample();
            //new HOGSample();
            //new HoughLinesSample();
            //new KAZESample2();
            //new KAZESample();
            //new MatOperations();
            //new MatToBitmap();
            //new MDS();
            //new MSERSample();
            //new NormalArrayOperations();
            //new PhotoMethods();
            //new MergeSplitSample();
            //new MorphologySample();
            //new PixelAccess();
            //new Pose();
            //new SeamlessClone();
            //new SiftSurfSample();
            //new SimpleBlobDetectorSample();
            //new SolveEquation();
            //new StarDetectorSample();
            //new Stitching();
            //new Subdiv2DSample();
            //new SuperResolutionSample();
            //new SVMSample();
            //new VideoWriterSample();
            //new VideoCaptureSample();

            sample.Run();
        }
    }
}
