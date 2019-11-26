using OpenCvSharp;
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
            var lenna = new Mat(FilePath.Image.Zhangly);
            var lenna511 = new Mat(FilePath.Image.Zhangly1);
            var s = Similar(ref lenna, ref lenna511);
            Console.WriteLine($"相似度：{s}");

            // Load the pictures
            var haarCascade = new CascadeClassifier(FilePath.Text.HaarCascade);

            var lennaResult = this.DetectFace(lenna, haarCascade);
            var lenna511Result = this.DetectFace(lenna511, haarCascade);

            Cv2.ImShow(FilePath.Image.Lenna, lennaResult);
            Cv2.ImShow(FilePath.Image.Lenna511, lenna511Result);

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
        private float Similar(ref Mat src, ref Mat src2)
        {
            Mat gray1 = new Mat(src.Size(), src.Type()),
                gray2 = new Mat(src2.Size(), src2.Type());
            Cv2.CvtColor(src, gray1, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(src2, gray2, ColorConversionCodes.BGR2GRAY);

            var size = new Size(512, 512);
            using (var scaledImg1 = gray1.Resize(size))
            using (var scaledImg2 = gray2.Resize(size))
            {
                Cv2.Threshold(scaledImg1, scaledImg1, 128, 255, ThresholdTypes.BinaryInv);
                Cv2.Threshold(scaledImg2, scaledImg2, 128, 255, ThresholdTypes.BinaryInv);
                Mat res = new Mat(size, scaledImg1.Type());
                Cv2.Absdiff(scaledImg1, scaledImg2, res);
                //Cv2.ImShow("aa", scaledImg1);
                var all = (float)scaledImg1.Sum();
                var result = (float)res.Sum();
                return (1 - result / all);
            }
        }

        private Mat DetectFace(Mat src, CascadeClassifier cascade)
        {
            Mat result;

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