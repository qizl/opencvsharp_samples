using OpenCvSharp;
using OpenCvSharp.XFeatures2D;
using SampleBase;
using System;

namespace SamplesCS
{
    /// <summary>
    /// Human face detection
    /// http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
    /// </summary>
    class SimilarCompare : ISample
    {
        public void Run()
        {
            //using var src1 = new Mat(FilePath.Image.Zhangly, ImreadModes.Color);
            //using var src2 = new Mat(FilePath.Image.Zhangly3, ImreadModes.Color);

            //MatchBySift(src1, src2);

            var haarCascade = new CascadeClassifier(FilePath.Text.HaarCascade);
            
            var lennaResult = this.DetectFace(FilePath.Image.Zhangly, haarCascade);
            var lenna511Result = this.DetectFace(FilePath.Image.Zhangly1, haarCascade);

            Cv2.ImShow(FilePath.Image.Lenna, lennaResult);
            Cv2.ImShow(FilePath.Image.Lenna511, lenna511Result);

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        private Mat DetectFace(string fileName, CascadeClassifier cascade)
        {
            Mat result;
            using (var src = new Mat(fileName))
            using (var gray = new Mat())
            {
                result = src.Clone();
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

                // Detect faces
                Rect[] faces = cascade.DetectMultiScale(gray, 1.08, 2, HaarDetectionType.ScaleImage, new Size(30, 30));

                // Render all detected faces
                foreach (Rect face in faces)
                {
                    var center = new Point
                    {
                        X = (int)(face.X + face.Width * 0.5),
                        Y = (int)(face.Y + face.Height * 0.5)
                    };
                    var axes = new Size
                    {
                        Width = (int)(face.Width * 0.5),
                        Height = (int)(face.Height * 0.5)
                    };
                    Cv2.Ellipse(result, center, axes, 0, 0, 360, new Scalar(255, 0, 255), 4);
                }
            }
            return result;
        }

        
    }
}