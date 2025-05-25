#pragma once

#include "cppTypes.h"

class OrientationTools {
public:
    OrientationTools() {}
    ~OrientationTools() {}
    
    Matrix3d coordinateRotation(const CoordinateAxis& axis, const double& theta){
    double s = std::sin(theta);
    double c = std::cos(theta);
    Matrix3d R = Matrix3d::Identity();

    if (axis == CoordinateAxis::X) {
      R << 1, 0, 0, 0, c, -s, 0, s, c;
    } else if (axis == CoordinateAxis::Y) {
      R << c, 0, s, 0, 1, 0, -s, 0, c;
    } else if(axis == CoordinateAxis::Z) {
      R << c, -s, 0, s, c, 0, 0, 0, 1;
    } else {
      std::cout << "unknown coordinate axis, skip cal rotation matrix\n";
    }
    
    return R;
  }

    int sgn(double a) {
        if(a>=0) {
            return 1;
        } else {
            return -1;
        }
    }

    /*******************************************basic transform*******************************************************/
    Matrix3d vecToSkew(const Vector3d& v){
        Matrix3d temp;
        temp << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return temp;
    }
    Vector3d skewToVec(const MatrixXd& M){
        Vector3d temp;
        temp << M(2, 1), M(0, 2), M(1, 0);
        return temp;
    }
    Vector3d eulerAngleRateToAngVelWorld(const Vector3d& rpyAngle, const Vector3d& rpyAngleRate){
        Matrix3d R;
        Vector3d angvel;
        double spitch = std::sin(rpyAngle[1]);
        double cpitch = std::cos(rpyAngle[1]);
        double syaw = std::sin(rpyAngle[2]);
        double cyaw = std::cos(rpyAngle[2]);
        R << cpitch * cyaw, -syaw, 0,
            cpitch * syaw, cyaw, 0,
            -spitch, 0, 1;
        angvel = R * rpyAngleRate;

        return angvel;
    }
    Matrix3d rpyrateToAngvelWorldMat(const Vector3d& rpy) {
        Matrix3d R;
        R << cos(rpy[1]) * cos(rpy[2]), -sin(rpy[2]), 0,
            cos(rpy[1]) * sin(rpy[2]), cos(rpy[2]), 0,
            -sin(rpy[1]), 0, 1;

        return R;
    }
    Matrix3d eulerAngleToRotMatrix(const Vector3d& rpyAngle){
        AngleAxisd rollAngle(AngleAxisd(rpyAngle(0),Vector3d::UnitX()));
        AngleAxisd pitchAngle(AngleAxisd(rpyAngle(1),Vector3d::UnitY()));
        AngleAxisd yawAngle(AngleAxisd(rpyAngle(2),Vector3d::UnitZ()));
        Matrix3d rot;
        rot = yawAngle * pitchAngle * rollAngle;//ZYX
        return rot;
    }
    Vector3d rotMatrixToEulerAngle(const Matrix3d& rot){
        Quaterniond temp(rot);
        temp.normalize();
        Vector3d euler;
        euler = temp.matrix().eulerAngles(2,1,0);//ZYX -> ypr
        return euler.reverse();//rpy
    }
    Quaterniond eulerAngleToQuat(const Vector3d& rpyAngle){
        Matrix3d rot;
        rot = eulerAngleToRotMatrix(rpyAngle);
        Quaterniond quat(rot);
        return quat;
    }
    // Vector3d quatToEulerAngle(const Quaterniond& quat){
    //     Vector3d euler;
    //     Quaterniond temp = quat.normalized();
    //     Quaterniond quat_zero(1, 0, 0, 0);
    //     // if (temp.dot(quat_zero) < 0) {
    //     //     temp.w() *= -1;
    //     //     temp.vec() *= -1;
    //     // }
    //     euler = temp.matrix().eulerAngles(2,1,0);//ZYX -> ypr
    //     return euler.reverse();//rpy
    // }
    Vector3d quatToEulerAngle(const Quaterniond& quat) {
        Vector3d rpy;
        Vector<double, 4> q;
        auto square = [](double i) {return i * i;};
        q[0] = quat.w();
        q.tail(3) = quat.vec();
        double as = std::min(-2. * (q[1] * q[3] - q[0] * q[2]), .99999);
        rpy(2) =
            std::atan2(2 * (q[1] * q[2] + q[0] * q[3]),
                        square(q[0]) + square(q[1]) - square(q[2]) - square(q[3]));
        rpy(1) = std::asin(as);
        rpy(0) =
            std::atan2(2 * (q[2] * q[3] + q[0] * q[1]),
                        square(q[0]) - square(q[1]) - square(q[2]) + square(q[3]));
        return rpy;
    }
    Matrix3d quatToRotMatrix(const Quaterniond& quat){
        Matrix3d rot;
        Quaterniond temp = quat.normalized();
        rot = temp.matrix();
        return rot;
    }

    /*******************************************basic calculation*****************************************************/
    Quaterniond quatMultiply(const Quaterniond& q1, const Quaterniond& q2){
        Matrix<double, 4, 4> leftMulti;
        Vector<double, 4> quat2, temp;
        leftMulti = q1.w() * Eigen::MatrixXd::Identity(4, 4);
        leftMulti.bottomLeftCorner(3,1) = q1.vec();
        leftMulti.topRightCorner(1,3) = -q1.vec().transpose();
        leftMulti.bottomRightCorner(3,3) += vecToSkew(q1.vec());
        quat2 << q2.w(), q2.vec();
        temp = leftMulti * quat2;
        Quaterniond ans(temp(0), temp(1), temp(2), temp(3));
        return ans;
    }
    Quaterniond quatInverse(const Quaterniond& quat){
        double norm = quat.norm();
        Vector<double, 4> temp;
        temp << quat.w(), -quat.vec();
        temp /= norm;
        Quaterniond ans(temp(0), temp(1), temp(2), temp(3));
        return ans;
    }

    /******************************************SO(3) && so(3)**********************************/
    //so(3) -> SO(3)
    Quaterniond exp(const Vector3d& rot_vec){
        double temp = rot_vec.norm() / 2.0;
        Quaterniond quat;
        quat.w() = std::cos(temp);
        quat.vec() = std::sin(temp) * rot_vec.normalized();
        return quat;
    }
    //Rodrigues' Rotation Formula
    Matrix3d expRod(const Vector3d& rot_vec){
        Matrix3d rot;
        double theta = rot_vec.norm();
        Vector3d r = rot_vec.normalized();
        rot = std::cos(theta) * Matrix3d::Identity() +
            (1 - std::cos(theta)) * r * r.transpose() +
            std::sin(theta) * vecToSkew(r);
        return rot;
    }
    //SO(3) -> so(3)
    Vector3d log(const Matrix3d& R){
        Vector3d rot_vec;
        double theta = std::acos((R.trace() - 1.0) / 2.0);
        double coefficient = (std::abs(theta) >= 1e-6) ? theta / (2.0 * std::sin(theta)) : 0.5;
        Matrix3d temp;
        temp = coefficient * (R - R.transpose());
        return skewToVec(temp);
    }
    Vector3d log(const Quaterniond& quat){
        Vector3d rot_vec;
        Quaterniond temp = quat.normalized();
        if(std::abs(1.0-temp.w()) < 0.001){
            rot_vec = 2.0 * temp.vec();
        }
        else{
            double rot_vec_norm = 2 * std::acos(temp.w());
            rot_vec = rot_vec_norm * temp.vec() / (std::sin(rot_vec_norm / 2.0));
        }
        return rot_vec;
    }
    //orientation err
    Quaterniond quatError(const Quaterniond& quatDesired, Quaterniond quatCurr){
        //q && -q represents the same rotation. Check dot < 0? To keep the distance as short as possible
        if(quatCurr.dot(quatDesired) < 0) {
            quatCurr.w() *= -1.0;
            quatCurr.vec() *= -1.0;
        }
        Quaterniond err = quatDesired * quatCurr.inverse();
        return err;
    }

    //#############################################################################################
    Quaterniond rotationMatrixToQuaternion(const Mat3<double>& r1) {
        Quaterniond q;

        double tr = r1.trace();
        double S = sqrt(tr + 1.0);

        q.w() = 0.5 * sqrt(tr + 1);
        q.x() = 0.5 * sgn(r1(2, 1) - r1(1, 2)) * sqrt(r1(0, 0) - r1(1, 1) - r1(2, 2) + 1);
        q.y() = 0.5 * sgn(r1(0, 2) - r1(2, 0)) * sqrt(r1(1, 1) - r1(2, 2) - r1(0, 0) + 1);
        q.z() = 0.5 * sgn(r1(1, 0) - r1(0, 1)) * sqrt(r1(2, 2) - r1(0, 0) - r1(1, 1) + 1);

        return q;
    }

    Vec3<double> rotationMatrixToRPY(const Mat3<double>& R) {
        Quaterniond q = rotationMatrixToQuaternion(R);
        Quaterniond quat;
        Vec3<double> rpy = quatToEulerAngle(q);
        return rpy;
    }
};

