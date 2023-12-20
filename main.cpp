#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <condition_variable>
#include "boost/filesystem.hpp"

class Detector
{
private:
    cv::String m_WindowsName{"Detector"};
    std::vector<cv::Mat>m_Images;
    int m_DelayForKey{1};

    cv::dnn::Net m_opticalModel;
    cv::dnn::Net m_infraredModel;


    std::mutex m_ActivityMutex;
    std::chrono::seconds m_TimeOutActivity{std::chrono::seconds(10)};
    std::condition_variable m_ActivityConditionVariable;

    std::thread m_WorkThread;

    bool m_IsWorking{false};

    static std::vector<cv::Mat> GetImagesFromDirectory()
    {
        std::string pathToDirectory = "../images";
        std::string fileExtension = ".png";
        boost::filesystem::path directory_path{pathToDirectory};

        boost::filesystem::directory_iterator end_iterator;
        std::vector<cv::Mat>images;

        if (!boost::filesystem::exists(directory_path) || !boost::filesystem::is_directory(directory_path))
        {
            std::cerr << "Invalid directory: " << pathToDirectory << ". EXIT" << std::endl;
            exit(EXIT_FAILURE);
        }

        for (boost::filesystem::directory_iterator it(directory_path); it != end_iterator; ++it)
        {
            if (boost::filesystem::is_regular_file(it->status()) && boost::filesystem::extension(it->path()) == fileExtension)
            {
                cv::Mat image{cv::imread(it->path().string())};

                if(image.empty())
                {
                    std::cerr << "Failed to load image from " << pathToDirectory << " directory" << std::endl;
                    std::cerr << "File name: " << it->path().string() << std::endl;
                    continue;
                }

                std::cout << "File " << it->path().string() << " successfully loaded" << std::endl;

                images.push_back(image);
            }
        }

        std::cout << images.size() << " images was uploaded" << std::endl;

        return images;
    }

    static cv::dnn::Net GetOpticalModel()
    {
        std::string weights_path{"../YOLO/yolov7-tiny.weights"};
        std::string config_path{"../YOLO/yolov7-tiny.cfg"};

//        std::string weights_path{"../YOLO/yolov5m-seg.pt"};
//        std::string config_path{"../YOLO/yolov5n.yaml"};


        cv::dnn::Net net = cv::dnn::readNetFromDarknet(config_path, weights_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        if(net.empty())
        {
            std::cerr << "NO LAYERS IN MODEL. EXIT" << std::endl;
            exit(EXIT_FAILURE);
        }
        return net;
    }


    void ActivitySensor()
    {
        std::unique_lock<std::mutex>unique_lock(m_ActivityMutex);
        m_ActivityConditionVariable.notify_all();
    }

    void Work()
    {
        if(m_Images.empty())
        {
            std::cerr << "No photos to work with " << std::endl;
            return;
        }

        auto image = m_Images.begin();

        while(m_IsWorking)
        {
            ActivitySensor();

            if(image->empty())
            {
                std::cerr << "Image is empty" << std::endl;
                continue;
            }

            FindPeople(*image);
            FindCars(*image);

            cv::imshow(m_WindowsName, *image);

            int key_code = cv::waitKey(m_DelayForKey);

            if(image != m_Images.begin() && (key_code == 'a' || key_code == 'A'))
                image--;

            else if(image != m_Images.end() - 1 && (key_code == 'd' || key_code == 'D'))
                image++;

            else if(key_code == ' ')
            {
                m_IsWorking = false;
            }
        }
    }

    void FindCars(cv::Mat& image)
    {
        float confidence_threshold = 0.1;
        float nms_threshold = 0.3;

        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0,0, 0),true, false);
        m_opticalModel.setInput(blob);

        std::vector<cv::Mat>outs;

        m_opticalModel.forward(outs, m_opticalModel.getUnconnectedOutLayersNames());

        std::vector<float>confidences;
        std::vector<cv::Rect>boxes;

        for(const auto & out : outs)
        {
            for (int i = 0; i < out.rows; ++i)
            {
                cv::Mat detection = out.row(i);
                cv::Mat scores = detection.colRange(5, detection.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

                if (confidence > confidence_threshold)
                {
                    int centerX = static_cast<int>(detection.at<float>(0, 0) * image.cols);
                    int centerY = static_cast<int>(detection.at<float>(0, 1) * image.rows);
                    int width = static_cast<int>(detection.at<float>(0, 2) * image.cols);
                    int height = static_cast<int>(detection.at<float>(0, 3) * image.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    confidences.push_back(static_cast<float>(confidence));
                    boxes.emplace_back(left, top, width, height);
                }

            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

        for (int i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            // Draw bounding box.
            rectangle(image, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(255,178,50), 3 * 1);
            // Get the label for the class name and its confidence.
//            std::string label = format("%.2f", confidences[idx]);
//            label = class_name[class_ids[idx]] + ":" + label;
//            // raw class labels.D
//            draw_label(input_image, label, left, top);
        }

        for (int idx : indices)
            rectangle(image, boxes[idx], cv::Scalar(255, 178, 50), 2);
    }

    void FindPeople(const cv::Mat& image)
    {
        cv::HOGDescriptor hog_descriptor;
        std::vector<cv::Rect>people;

        float hit_threshold = 0.0f;
        float scale_factor = 1.059f;
        float group_threshold = 2.0f;

        hog_descriptor.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

        hog_descriptor.detectMultiScale(image, people, hit_threshold, cv::Size{8, 8}, cv::Size{32, 32}, scale_factor, group_threshold);

        for(auto& person : people)
        {
            person.x += cvRound(person.width * 0.5);

            person.x += -40;
            person.width = cvRound(person.width * 0.3);
            person.y += cvRound(person.height * 0.07);
            person.height = cvRound(person.height * 0.8);
            rectangle(image, person.tl(), person.br(), cv::Scalar(0,255,0), 3);
        }
    }

public:
    explicit Detector(int timeOutActivity = 10) :
            m_TimeOutActivity{std::chrono::seconds(timeOutActivity)}{}

    void Start()
    {
        m_Images = GetImagesFromDirectory();
        m_opticalModel = GetOpticalModel();
        //m_infraredModel = GetInfraredModel();
        m_IsWorking = true;

        m_WorkThread = std::thread([this]{Work();});
        m_WorkThread.detach();
    }

    bool IsActive()
    {
        std::unique_lock<std::mutex>unique_lock(m_ActivityMutex);

        if(!m_IsWorking) return false;

        return m_ActivityConditionVariable.wait_for(unique_lock, m_TimeOutActivity) == std::cv_status::no_timeout;
    }
};

int main()
{
    Detector detector{};

    detector.Start();

    while(true)
    {
        if(!detector.IsActive())
        {
            std::cerr << "No respond from detector. EXIT" << std::endl;
            cv::destroyAllWindows();
            exit(EXIT_FAILURE);
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}
