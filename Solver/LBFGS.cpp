#include "LBFGS.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>

template<typename Scalar>
LBFGSCompactB<Scalar>::LBFGSCompactB(const std::deque<Vector> &m_s, const std::deque<Vector> &m_y, Scalar sigmak)
	:m_sigma_k(sigmak)
{
	if (m_s.size() == 0)
	{
		m_m = 0;
		return;
	}

	m_m = (int)m_s.size();

	S_k.resize(m_s[0].size(), m_m);
	Y_k.resize(m_s[0].size(), m_m);
	Vector d(m_m);
	L_k.resize(m_m, m_m);

	for (int i = 0; i < m_m; i++)
	{
		Y_k.col(i) = m_y[i];
		S_k.col(i) = m_s[i];
		d[i] = m_s[i].dot(m_y[i]);

		for (int j = 0; j < m_m; j++)
		{
			if (i > j)
			{
				L_k(i, j) = m_s[i].dot(m_y[j]);
			}
			else
			{
				L_k(i, j) = Scalar(0.0);
			}
		}
	}

	dhalf = d.cwiseSqrt();
	dhalfInv = dhalf.cwiseInverse();

	Matrix M = m_sigma_k * S_k.transpose() * S_k + L_k * d.cwiseInverse().asDiagonal() * L_k.transpose(); //(Eq.2.26)
	m_JkJkT.compute(M);
}

template<typename Scalar>
typename LBFGSCompactB<Scalar>::RowVector LBFGSCompactB<Scalar>::getWRow(int idx) const {
	RowVector result(2 * m_m);
	result << Y_k.row(idx), (m_sigma_k * S_k.row(idx));
	return result;
}
template<typename Scalar>
typename LBFGSCompactB<Scalar>::Vector LBFGSCompactB<Scalar>::applyWTranposed(const Vector &v) const {
	Vector p0 = Y_k.transpose() * v;
	Vector p1 = m_sigma_k * (S_k.transpose() * v);
	Vector result(p0.size() + p1.size());
	result << p0, p1;
	return result;
}
template<typename Scalar>
typename LBFGSCompactB<Scalar>::Vector LBFGSCompactB<Scalar>::applyM(const Vector &v) const {

	assert(v.size() == 2 * m_m);
	Vector p0 = v.head(m_m);
	Vector p1 = v.tail(m_m);

	// Step 6.1
	// [ D_k^{1/2},           0] | p2 | = |p0|
	// [ - L_k D_k^{-1/2], J_k ] | p3 |   |p1|
	//Vector p2 = dhalfInv.cwiseProduct(p0);
	Vector p2 = p0.cwiseQuotient(dhalf);
	//Vector p3 = m_JkJkT.matrixL().solve(p1 + L_k * (dhalfInv.cwiseProduct(p2)));
	Vector p3 = m_JkJkT.matrixL().solve(p1 + L_k * (p2.cwiseQuotient(dhalf)));
	// Step 6.2
	// [ -D_k^{1/2}, D_k^{-1/2} L_k^T] | p4 | = [ p2 ]
	// [ 0 ,                   J_k^T ] | p5 | = [ p3 ]
	// => - D_k^{1/2} p4 = p2 - D_k^{-1/2} L_k^T p5
	// => p4 = D_k^{-1/2} * ( D_k^{-1/2} L_k^T p5 - p2)
	Vector p5 = m_JkJkT.matrixU().solve(p3);
	//Vector p4 = dhalfInv.cwiseProduct(dhalfInv.cwiseProduct(L_k.transpose() * p5) - p2);
	Vector p4 = ((L_k.transpose() * p5).cwiseQuotient(dhalf) - p2).cwiseQuotient(dhalf);

	Vector result(p4.size() + p5.size());
	result << p4, p5;
	return result;
}

template<typename Scalar>
typename LBFGSCompactB<Scalar>::Vector LBFGSCompactB<Scalar>::operator*(const Vector &v) const
{
	if (m_m == 0)
	{
		return m_sigma_k * v;
	}

	//Alg 3.2
	// Step 5
	Vector p0 = Y_k.transpose() * v;
	Vector p1 = m_sigma_k * (S_k.transpose() * v);

	// Step 6.1
	// [ D_k^{1/2},           0] | p2 | = |p0|
	// [ - L_k D_k^{-1/2], J_k ] | p3 |   |p1|
	//Vector p2 = dhalfInv.cwiseProduct(p0);
	Vector p2 = p0.cwiseQuotient(dhalf);
	//Vector p3 = m_JkJkT.matrixL().solve(p1 + L_k * (dhalfInv.cwiseProduct(p2)));
	Vector p3 = m_JkJkT.matrixL().solve(p1 + L_k * (p2.cwiseQuotient(dhalf)));
	// Step 6.2
	// [ -D_k^{1/2}, D_k^{-1/2} L_k^T] | p4 | = [ p2 ]
	// [ 0 ,                   J_k^T ] | p5 | = [ p3 ]
	// => - D_k^{1/2} p4 = p2 - D_k^{-1/2} L_k^T p5
	// => p4 = D_k^{-1/2} * ( D_k^{-1/2} L_k^T p5 - p2)
	Vector p5 = m_JkJkT.matrixU().solve(p3);
	//Vector p4 = dhalfInv.cwiseProduct(dhalfInv.cwiseProduct(L_k.transpose() * p5) - p2);
	Vector p4 = ((L_k.transpose() * p5).cwiseQuotient(dhalf) - p2).cwiseQuotient(dhalf);

	return m_sigma_k * v - Y_k * p4 - m_sigma_k * (S_k * p5);
}
template<typename Scalar>
typename LBFGSCompactB<Scalar>::Matrix LBFGSCompactB<Scalar>::computeDense(int d) const
{
	Matrix B(d, d);
	Vector e_i = Vector::Zero(d);
	for (int i = 0; i < d; i++)
	{
		e_i[i] = Scalar(1.0);
		B.col(i) = (*this).operator*(e_i);
		e_i[i] = Scalar(0.0);
	}
	return B;
}


template<typename Scalar>
LBFGS<Scalar>::LBFGS()
	:m_m(20),
	m_updateCount(0)
{
}
template<typename Scalar>
LBFGS<Scalar>::LBFGS(int m)
	:m_m(m),
	m_updateCount(0)
{
}
template<typename Scalar>
int LBFGS<Scalar>::getHistorySize() const {
	return m_m;
}
template<typename Scalar>
void LBFGS<Scalar>::setHistorySize(int m)
{
	m_m = m;
	while (m_s.size() > m_m)
	{
		m_rho.pop_front();
		m_s.pop_front();
		m_y.pop_front();
	}
}

template<typename Scalar>
void LBFGS<Scalar>::apply(Vector &d, std::function<void(Vector &v)> applyH0) const
{
	Vector alpha(m_m);
	for (int i = (int)m_rho.size() - 1; i >= 0; i--)
	{
		Scalar alpha_i = m_rho[i] * m_s[i].dot(d);
		alpha[i] = alpha_i;
		d -= alpha_i * m_y[i];
	}

	applyH0(d);

	for (int i = 0; i < (int)m_rho.size(); i++)
	{
		Scalar beta = m_rho[i] * m_y[i].dot(d);
		d += m_s[i] * (alpha[i] - beta);
	}
}
template<typename Scalar>
void LBFGS<Scalar>::apply(Vector &d, const Vector &z, std::function<void(Vector &v)> applyH0) const {
	for (int i = 0; i < d.size(); i++) {
		if (z[i] == 0.0 && d[i] != 0.0) {
			throw std::invalid_argument("invalid arg");
		}
	}

	//no line search can make sure s'*y is positive for any kind of active set...
	// so lets add a check
	std::vector<bool> applied(m_y.size());
	std::vector<Scalar> rhoZ(m_y.size());

	Vector alpha(m_m);
	for (int i = (int)m_y.size() - 1; i >= 0; i--)
	{
		Scalar invRho = m_s[i].dot(z.cwiseProduct(m_y[i]));
		if (invRho > 0.0) {
			applied[i] = true;
			rhoZ[i] = 1.0 / invRho;
			Scalar alpha_i = rhoZ[i] * m_s[i].dot(d); // dont need to project here, cause d has zero in the correct positions
			alpha[i] = alpha_i;
			d -= alpha_i * (z.cwiseProduct(m_y[i]));
		}
		else {
			std::cout << " m_s[i].dot(m_y[i]) = " << m_s[i].dot(m_y[i]) << std::endl;
			std::cout << " invRho  < 0.0 " << invRho << " , skipping " << i << std::endl;
			applied[i] = false;
		}
	}
	for (int i = 0; i < d.size(); i++) {
		if (z[i] == 0.0 && d[i] != 0.0) {
			throw std::logic_error("bug");
		}
	}

	applyH0(d);

	for (int i = 0; i < d.size(); i++) {
		if (z[i] == 0.0 && d[i] != 0.0) {
			throw std::invalid_argument("invalid arg");
		}
	}

	for (int i = 0; i < (int)m_y.size(); i++)
	{
		if (applied[i]) {
			Scalar beta = rhoZ[i] * m_y[i].dot(d); // dont need to project here, cause d has zero in the correct positions but you might want to add an assert?
			d += z.cwiseProduct(m_s[i]) * (alpha[i] - beta);
		}
	}

	for (int i = 0; i < d.size(); i++) {
		if (z[i] == 0.0 && d[i] != 0.0) {
			throw std::logic_error("bug");
		}
	}
}
template<typename Scalar>
Scalar LBFGS<Scalar>::computeDefaultScaling() const
{
	Scalar scaling = m_s.back().dot(m_y.back()) / m_y.back().squaredNorm();
	return scaling;
}
template<typename Scalar>
Scalar LBFGS<Scalar>::computeDefaultScaling(std::function<void(Vector &v)> applyH0) const
{
	Vector H0y = m_y.back();
	applyH0(H0y);
	Scalar scaling = m_s.back().dot(m_y.back()) / m_y.back().dot(H0y); // see "updating Quasi-Newton matrices with Limited Storage" page 781
	return scaling;
}
template<typename Scalar>
Scalar LBFGS<Scalar>::computeDefaultScaling(const Vector &z, std::function<void(Vector &v)> applyH0) const
{
	int largestAppliedIdx;
	for (largestAppliedIdx = m_s.size() - 1; largestAppliedIdx >= 0; largestAppliedIdx--) {
		Scalar invrho = m_s[largestAppliedIdx].dot(z.cwiseProduct(m_y[largestAppliedIdx]));
		if (invrho > 0.0) break;
	}
	if (largestAppliedIdx == -1) return 1.0;

	const Vector &s_k = m_s[largestAppliedIdx];
	const Vector &y_k = m_y[largestAppliedIdx];
	Vector H0y = z.cwiseProduct(y_k);
	Scalar skyk = s_k.dot(z.cwiseProduct(y_k)); // We are assuming z[i] is either 0 or 1
	applyH0(H0y);
	for (int i = 0; i < H0y.size(); i++) {
		if (z[i] == 0.0 && H0y[i] != 0.0) throw std::invalid_argument("invalid arg");
	}
	Scalar denominator = (y_k.cwiseProduct(z)).dot(H0y);
	std::cout << " largestAppliedIdx " << largestAppliedIdx << std::endl;
	std::cout << " skyk " << skyk << std::endl;
	std::cout << " denominator " << denominator << std::endl;
	//std::cout << " skyk " << skyk << std::endl;
	//std::cout << " denonminator " << denominator << std::endl;
	//std::cout << " z " << z << std::endl;
	//std::cout << " m_y.size() " << m_y.size() << std::endl;
	Scalar scaling = skyk / denominator; // see "updating Quasi-Newton matrices with Limited Storage" page 781
	return scaling;
}

template<typename Scalar>
void LBFGS<Scalar>::apply(Vector &d, Scalar firstStepSize) const {
	if (m_s.size() == 0)
	{
		//this->apply(d, [](Vector &d) {});
		// use identity as initial approximation

		d *= firstStepSize / d.norm();
	}
	else
	{
		this->apply(d, [this](Vector &d0) {
			Scalar f = this->computeDefaultScaling();
			d0 *= f;
		});
	}
}
template<typename Scalar>
void LBFGS<Scalar>::apply(Vector &d) const
{
	this->apply(d, Scalar(1.0));
}
template<typename Scalar>
void LBFGS<Scalar>::apply(Vector &d, const Vector &z, Scalar firstStepSize) const {
	if (m_s.size() == 0)
	{
		//this->apply(d, [](Vector &d) {});
		// use identity as initial approximation

		d *= firstStepSize / d.norm();
	}
	else
	{
		this->apply(d, z, [this, &z](Vector &d0) {
			auto identity = [](Vector &v) {};
			Scalar f = this->computeDefaultScaling(z, identity);
			std::cout << " scaling " << f << std::endl;
			d0 *= f;
		});
	}
}


template<typename Scalar>
LBFGSCompactB<Scalar> LBFGS<Scalar>::computeCompactB(Scalar sigmak) const
{
	return LBFGSCompactB<Scalar>(m_s, m_y, sigmak);
}
template<typename Scalar>
typename LBFGS<Scalar>::Matrix LBFGS<Scalar>::computeB(int d, Scalar sigmak) const
{
	Matrix B = sigmak * Matrix::Identity(d, d);

	for (int i = 0; i < m_y.size(); i++)
	{
		Vector Bs = B * m_s[i];
		B += m_y[i] / (m_y[i].dot(m_s[i]))  * m_y[i].transpose() + Bs / (m_s[i].dot(Bs)) * Bs.transpose();
	}

	return B;
}
template<typename Scalar>
void LBFGS<Scalar>::computeCompactB(Scalar theta, Matrix &W, Matrix &M) {
	int m = (int)m_y.size();

	if (m == 0) {
		W = Matrix(m_y[0].size(), 2 * m);
		M = Matrix();
		return;
	}

	W.resize(m_y[0].size(), 2 * m);

	Matrix Minv(2 * m, 2 * m);
	for (int i = 0; i < m; i++)
	{
		W.col(i) = m_y[i];
		W.col(i + m) = theta * m_s[i];
		Scalar di = m_s[i].dot(m_y[i]);

		Minv(i, i) = -di;

		for (int j = 0; j < m; j++)
		{
			if (i > j)
			{
				//L_k(i, j) = m_s[i].dot(m_y[j]);
				Scalar siyj = m_s[i].dot(m_y[j]);
				Minv(m + i, j) = siyj;
				Minv(j, m + i) = siyj;
			}
			else
			{
				//L_k(i, j) = Scalar(0.0);
				Minv(m + i, j) = Scalar(0.0);
				Minv(j, m + i) = Scalar(0.0);
			}

			Minv(m + i, m + j) = theta * ( m_s[i].dot(m_s[j]) );
		}
	}
	M = Minv.inverse();
}

template<typename Scalar>
void LBFGS<Scalar>::update(const Vector &s, const Vector &y)
{
	using std::abs;
	m_updateCount += 1;

	Scalar invrho = s.dot(y);
	//if (abs(invrho) < 1e-16)
	//if (invrho < 1e-16 * y.squaredNorm())
	if (invrho / s.squaredNorm() < 1e-10)
	{
		std::cout << " LBFGS s.dot(y) " << invrho << std::endl;
		std::cout << " skipping update " << std::endl;
		return;
	}

	m_rho.push_back(Scalar(1)/ invrho);
	m_s.push_back(s);
	m_y.push_back(y);
	if (m_s.size() > m_m)
	{
		m_rho.pop_front();
		m_s.pop_front();
		m_y.pop_front();
	}
}
template<typename Scalar>
void LBFGS<Scalar>::updateDamped(const Vector &s, const Vector &y)
{
	using std::min;

	Scalar scale = Scalar(1.0);
	if (m_s.size() > 0) scale = this->computeDefaultScaling();
	LBFGSCompactB<Scalar> B_k = computeCompactB(Scalar(1.0) / scale);
	//TODO add correct B_0
	Vector B_ks = B_k * s;
	Scalar sBs = s.dot(B_ks);
	Scalar skyk = y.dot(s);
	Scalar theta_k = (skyk >= Scalar(0.2) * sBs ? Scalar(1.0) : Scalar(0.8) * sBs / (sBs - skyk));

	Vector y_tilde = theta_k * y + (Scalar(1.0) - theta_k) * B_ks;

	this->update(s, y_tilde);
}
template<typename Scalar>
void LBFGS<Scalar>::updateDampedAlBaali2014(const Vector &s, const Vector &y)
{
	using std::min;

	//Damped Techniques for the Limited Memory BFGS
	// Method for Large - Scale Optimization  ( Al Baali 2014)
	//Scalar sigma1 = Scalar(0.999);
	Scalar sigma1 = Scalar(0.9);
	Scalar sigma2 = Scalar(0.6);
	Scalar sigma3 = Scalar(3.0);
	Scalar sigma4 = Scalar(0.0);
	
	Scalar scale = Scalar(1.0);
	if (m_s.size() > 0) scale = this->computeDefaultScaling();
	LBFGSCompactB<Scalar> B_k = computeCompactB(Scalar(1.0) / scale);
	Vector B_ks = B_k * s;
	Scalar sBks = s.dot(B_ks);
	if (sBks < Scalar(0.0)) std::cout << " error Bk already indefinite " << std::endl;
	Scalar tau_k = s.dot(y) / sBks;
	Vector H_ky = y;
	this->apply(H_ky);
	Scalar h_k = y.dot(H_ky) / s.dot(y);
	Scalar phi_k;
	if (tau_k <= Scalar(0.0) || (tau_k > Scalar(0.0) && tau_k <= min(Scalar(1.0) - sigma2, (Scalar(1.0) - sigma4) * h_k)))
	{
		phi_k = sigma2 / (Scalar(1.0) - tau_k);
	}
	else if (tau_k >= Scalar(1.0) + sigma3 && tau_k < (1.0 - sigma4) * h_k)
	{
		phi_k = sigma3 / (tau_k - Scalar(1.0));
	}
	else
	{
		phi_k = 1.0;
	}
	//std::cout << " phi_k " << phi_k << std::endl;

	Vector y_tilde = phi_k * y + (Scalar(1.0) - phi_k) * B_ks;

	this->update(s, y_tilde);
}

template<typename Scalar>
void LBFGS<Scalar>::clear()
{
	m_rho.clear();
	m_s.clear();
	m_y.clear();
	m_updateCount = 0;
}

template<typename Scalar>
int LBFGS<Scalar>::getUpdateCount() const
{
	return m_updateCount;
}

template<typename Scalar>
int LBFGS<Scalar>::getStoredUpdateCount() const
{
	return (int)m_s.size();
}
template<typename Scalar>
void LBFGS<Scalar>::popOldest()
{
	m_rho.pop_front();
	m_s.pop_front();
	m_y.pop_front();
}

template class LBFGSCompactB<float>;
template class LBFGSCompactB<double>;
template class LBFGS<float>;
template class LBFGS<double>;

