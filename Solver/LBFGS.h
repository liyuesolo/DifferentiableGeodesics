#ifndef LBFGS_H
#define LBFGS_H

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <deque>
#include <functional>


// Byrd 1994 Representations of quasi newton matrices (section 3.2)
//TODO support updating (which seems to be cheaper then computing L_k every time...)
template<typename Scalar>
class LBFGSCompactB
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	LBFGSCompactB(const std::deque<Vector> &m_s, const std::deque<Vector> &m_y, Scalar sigmak);

	RowVector getWRow(int idx) const;
	Vector applyWTranposed(const Vector &v) const;
	Vector applyM(const Vector &v) const;

	Vector operator*(const Vector &v) const;

	Matrix computeDense(int d) const;

	Scalar getSigma() const { return m_sigma_k; }
	int getVectorCount() const {
		return m_m;
	}


private:
	Scalar m_sigma_k;
	Matrix L_k;
	Matrix S_k, Y_k;
	Vector dhalfInv;
	Vector dhalf;
	Eigen::LLT<Matrix> m_JkJkT;
	int m_m;
};

template<typename Scalar>
class LBFGS
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	LBFGS();
	LBFGS(int m);

	void setHistorySize(int m);
	int getHistorySize() const;

	//this applies the inverse hessian approxmiation
	void apply(Vector &d, std::function<void(Vector &v)> applyH0) const;
	void apply(Vector &d, const Vector &z, std::function<void(Vector &v)> applyH0) const;
	Scalar computeDefaultScaling() const;
	Scalar computeDefaultScaling(std::function<void(Vector &v)> applyH0) const;
	Scalar computeDefaultScaling(const Vector &z, std::function<void(Vector &v)> applyH0) const;
	void apply(Vector &d, Scalar firstStepSize) const;
	void apply(Vector &d) const;
	void apply(Vector &d, const Vector &z, Scalar firstStepSize) const;

	//this applies the hessian approxmiation (NOT the inverse)
	LBFGSCompactB<Scalar> computeCompactB(Scalar sigmak) const;

	Matrix computeB(int d, Scalar sigmak) const;

	/** This computes the Hessian approximation I*theta - W*M*W' as described in
		"A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION"
		Eq. 3.3 and 3.4
		note that this is a direct version of the CompactB class but it uses .inverse which is not stable
	*/
	void computeCompactB(Scalar theta, Matrix &W, Matrix &M);

	void update(const Vector &s, const Vector &y);
	void updateDamped(const Vector &s, const Vector &y);
	void updateDampedAlBaali2014(const Vector &s, const Vector &y);

	void clear();

	int getUpdateCount() const;
	/** this returns the number of stored updates */
	int getStoredUpdateCount() const;

	void popOldest();

private:
	int m_m;
	int m_updateCount;
	std::deque<Scalar> m_rho;
	std::deque<Vector> m_s, m_y;
};

#endif