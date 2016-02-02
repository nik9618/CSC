#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include "omp.h"
#include <algorithm>
#include <iterator>

#define margin 100 // becareful margin must me at least 3*dsize
#define os(t,i) 	OS[t][i+margin]
#define s(t,i) 		S[t][i+margin]
#define d(t,k,i) 	D[t][k][i+dsize]
#define dm(k,i) 	DMASTER[k][i+dsize]
#define nd(t,k,i) 	ND[t][k][i+dsize]
#define z(t,k,i) 	Z[t][k][i+margin]
#define zn(t,k,i) 	Zn[t][k][i+margin]
#define b(t,k,i) 	B[t][k][i+margin]
#define LAMBDA		0.1
#define MAXDICITER 	1

#define T 24
#define K 40
#define dsize 4
#define ssize 1250
#define importDict true
using namespace std;

string resultpath = "/home/kanit/anomalydeep/result/";

double limitDiffLoss = 0.001;
int maxCoordinateDescentLoop = 1000;
int miniBatch = 5;
int maxData = 0;
double impDict[K*(dsize*2+1)] = {0.29151,0.291434,0.290295,0.288251,0.286535,0.284106,0.282069,0.278986,0.275104,0.264471,0.287212,0.291231,0.294471,0.294269,0.292423,0.288148,0.281679,0.273452,0.0234105,0.226943,0.560073,0.525201,0.263727,0.026853,-0.052284,-0.0517093,-0.0438268,0.275012,0.279263,0.281477,0.283707,0.285788,0.288084,0.289822,0.291756,0.293331,0.246467,0.282216,0.291949,0.298761,0.300933,0.299102,0.293758,0.282908,0.267977,0.204266,0.263612,0.300223,0.322169,0.326321,0.321964,0.306205,0.276487,0.228637,0.287877,0.516123,0.460323,0.286764,0.129587,0.0775422,0.0720986,0.054141,0.0455044,0.279957,0.283913,0.284497,0.285295,0.28589,0.286804,0.287113,0.287573,0.287808,-0.0652493,0.592991,0.537407,0.226644,-0.0153463,-0.0661842,-0.0654182,-0.0752962,-0.0880316,0.0296396,0.376043,0.597469,0.395513,0.134241,-0.109975,-0.087491,-0.0639627,-0.0728077,0.28116,0.286125,0.28606,0.286371,0.286353,0.286699,0.286159,0.285921,0.284065,0.286776,0.308474,0.314964,0.30977,0.297423,0.284301,0.270185,0.254118,0.230575,0.0105348,0.0104819,0.10519,0.120521,0.371799,0.372343,0.371043,0.373576,0.376332,0.306817,0.301664,0.2961,0.290623,0.285225,0.279969,0.274778,0.268839,0.261106,-0.0145129,0.00123815,0.0646206,0.364197,0.428101,0.442864,0.411076,0.0779021,0.0180235,0.281516,0.286328,0.286217,0.286384,0.286211,0.286501,0.286062,0.285848,0.283896,0.250318,0.263372,0.271436,0.278524,0.285499,0.292668,0.299543,0.306392,0.312668,0.134859,0.18424,0.205837,0.238495,0.277025,0.357754,0.40534,0.405121,0.168435,0.348494,0.345124,0.340449,0.331067,0.317718,0.296776,0.248028,0.108716,0.0323141,0.290562,0.289791,0.288691,0.287024,0.285959,0.284671,0.283855,0.281594,0.276375,0.0960347,-0.280198,-0.31249,0.305272,0.314603,0.28848,0.295272,0.253188,0.211486,0.343072,0.327192,0.313799,0.298759,0.284452,0.269326,0.253216,0.235727,0.213575,0.0136459,0.119442,0.451542,0.508212,0.447288,0.152519,0.0317562,-0.0137749,-0.00660277,0.0249111,0.0725156,0.126522,0.237739,0.378623,0.457847,0.422228,0.279221,0.123814,0.26687,0.273632,0.277889,0.281627,0.285633,0.289406,0.293368,0.297136,0.300932,0.279398,0.283059,0.283909,0.284919,0.28568,0.286793,0.287422,0.288451,0.289151,0.281616,0.286447,0.286285,0.286432,0.286266,0.286538,0.286001,0.285647,0.283734,0.296316,0.293861,0.290918,0.288238,0.285479,0.283082,0.280617,0.277759,0.271507,0.152882,0.213124,0.233004,0.26061,0.283758,0.307015,0.326099,0.348132,0.364702,0.287591,0.288214,0.28793,0.287246,0.286438,0.28591,0.285252,0.283613,0.276444,-0.029843,-0.0310456,-0.0120817,-0.116403,0.854667,0.250172,-0.0536816,-0.0243596,-0.0441034,0.299632,0.296628,0.293432,0.289297,0.285846,0.281658,0.278202,0.273661,0.268657,0.318937,0.311364,0.302986,0.291233,0.283628,0.274841,0.266909,0.260885,0.249121,0.297307,0.114292,0.127428,0.299109,0.302211,0.320557,0.308086,0.310266,0.29683,0.101101,0.604989,0.559827,0.119044,-0.0495776,-0.0842725,-0.0626136,-0.0706908,-0.0584508,-0.107736,-0.140649,-0.153362,-0.137112,0.00791108,0.473972,0.566307,0.359209,0.117044,0.277929,0.280891,0.283969,0.28372,0.286474,0.286354,0.288989,0.288849,0.291436,0.222134,0.246664,0.259003,0.271034,0.282952,0.295199,0.306859,0.319463,0.338658,0.343548,0.375002,0.341425,0.299669,0.258602,0.247467,0.22864,0.212865,0.203267,0.280348,0.286867,0.28756,0.288263,0.287878,0.28737,0.286113,0.284034,0.280294};

using namespace std;

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

vector< vector<double> > S(T,vector<double>(ssize+margin,0));
vector< vector<double> > OS(T,vector<double>(ssize+margin,0));
vector< vector< vector<double> > > D(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
vector< vector<double> > DMASTER(K,vector<double>(dsize*2+1,0));
vector< vector<double> > Zct(T,vector<double>(K,0));
vector< vector< vector<double> > > ND(T,vector< vector<double> >(K,vector<double>(dsize*2+1,0)));
vector< vector< vector<double> > > Z(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector< vector<double> > > Zn(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector< vector<double> > > B(T,vector< vector<double> >(K,vector<double>(ssize+3*margin,0)));
vector< vector<double> > ts;

double accumuZct(int t)
{
	for(int k=0;k<K;k++)
	{
		double ct = 0.f;
		for(int i = -dsize; i<ssize+dsize ;i++)
		{
			ct += fabs(z(t,k,i));
		}
		Zct[t][k] = (0.99)*Zct[t][k] + (0.01) * ct;
		cout << "Zct["<<k<<"]" << Zct[t][k]<<endl;
	}
}

double calcLoss(int t)
{
	double loss=0;
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(t,i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(t,k,j) * z(t,k,i+j);
			}
		}
		loss+= x*x;
	}
	return loss;
}

vector<double> reconstruct(int t)
{
	vector<double> res(ssize,0);
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(t,i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(t,k,j) * z(t,k,i+j);
			}
		}
		res[i] = x;
	}
//	if2
	return res;
}

void reportDZ(int t)
{
	vector<double> res = reconstruct(t);
	for(int i =0; i< ssize;i++)
	{
		cout << res[i] <<"\t";
	}
	cout<<endl;
}

void reportTesting(int t, string name)
{
	ofstream myfile;
	myfile.open (resultpath + name +".txt");
	
	// myfile << "OS" << "=[";	
	// for(int i = 0 ; i < ssize ; i++)
	// {
	// 	myfile << os(t,i)<<" ";
	// }
	// myfile << "]\n";
	myfile << "loss="<<calcLoss(t)<<"\n";
	myfile << "S" << "=[";	
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << s(t,i)<<" ";
	}
	myfile << "]\n";
	
	for(int k=0;k<K;k++)
	{
		myfile << "ND" << k << "=[";	
		for(int ds=-dsize;ds<=dsize; ds++)
		{
			myfile << nd(t,k,ds)<<" ";
		}
		myfile << "]\n";
	}

	vector<double> d = reconstruct(t);
	myfile << "DZ" << "=[";	
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << d[i] <<" ";
	}
	myfile << "]\n";

	myfile << "D" << "=[";
	for(int i = 0 ; i < K ; i++)
	{
		for (int j = -dsize; j<=dsize ;j++)
		{
			myfile << d(t,i,j)<<" ";
		}
		myfile << ";";
	}
	myfile << "]\n";
	myfile.close();
}

double lossDist(int t)
{
	double loss=0;
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(t,i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(t,k,j) * z(t,k,i+j);
			}
		}
		cout<< fabs(x)<<"\t";
	}
//	if2
	return loss;
}

void reportZ(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		cout << "Z"<<k<<":\t";
		for(int i = - dsize; i<= ssize + dsize ;i++)
		{
			cout << z(t,k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportD(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = - dsize; i<= dsize ;i++)
		{
			cout << d(t,k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportB(int t)
{
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			cout << b(t,k,i)<<"\t";
		}
		cout<<endl;
	}
}

void reportS(int t) {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
//	for(int i = 0 ; i < ssize;i++)s(i) = in[i];
	for(int i = 0; i < ssize;i++) cout << s(t,i) <<"\t";
	cout<<endl<<endl;
}

void initS(int t, vector<double> in) {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
	for(int i = 0 ; i < ssize;i++)
	{
		s(t,i) = in[i];
		os(t,i) = in[i];
	}

//	for(int i = -5 ; i < ssize+5 ;i++) cout << s(i) <<" ";
//	cout<<endl<<endl;
}

int validateS(int t)
{
	for(int i = 0 ; i < ssize;i++)
	{
		if(s(t,i) != s(t,i)) return 0;
	}
	return 1;
}

void normalizeS(int t)
{
	double sum = 0;
	double sum2 = 0;
	for(int i = 0 ; i < ssize;i++)
	{
		sum+=s(t,i);
		sum2 =(s(t,i)*s(t,i))+sum2;
	} 
	double mean = sum/ssize;
	double std = sqrt(sum2/ssize-mean*mean);
	if(std ==0) return;
	for(int i = 0 ; i < ssize;i++)
	{
		s(t,i) = ((s(t,i)-mean)/std);
		// s(t,i) = ((s(t,i)-mean));
	}
}

void initZ(int t) {
	//init Z
	for(int k=0;k<K;k++)
	{
		for(int i = -dsize;i<ssize+dsize;i++)
		{
			z(t,k,i) = 0;
		}
	}
}
void syncDictionary(int t)
{
	int anyNan = 0;
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = -dsize; i <=dsize ; i++ )
		{
			if(d(t,k,i) != d(t,k,i)) anyNan++;
		}
	}
	if(anyNan == 0)
	{
		printf(".");
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				for(int t = 0; t < T ; t++ )
				{
					dm(k,i) = 0.9*dm(k,i) + 0.1*d(t,k,i);
					d(t,k,i) = dm(k,i);
				}
			}
		}
	}
	else
	{
		printf("x");
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				for(int t = 0; t < T ; t++ )
				{
					d(t,k,i) = dm(k,i);
				}
			}
		}
	}
}

void normalizeDictionary(int t)
{
	for(int k = 0 ; k < K ; k++)
	{
		double total = 0.f;
		for(int i = -dsize; i <=dsize ; i++ )
		{
			total += d(t,k,i);
		}
		if(total != 0)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
				d(t,k,i) /= total;
			}
		}
	}
}

void importDictionary()
{
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			for(int t=0; t<T; t++)
			{
				d(t,k,i) = impDict[k*(dsize*2+1) + i+dsize];
			}
			dm(k,i) = impDict[k*(dsize*2+1) + i+dsize];
			printf("%f\t",d(0,k,i));
		}
	}
}

void initD()
{
	
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			// d(0,k,i) = 1./(1+2*dsize) + 1./(1+2*dsize)*(rand()%10000)/1000000.0;
			d(0,k,i) = 1./(1+2*dsize) + 1./(1+2*dsize)*(rand()%10000)/1000000.0;
		}
	}
	
	// normalizeDictionary(0);
	
	// duplicate them to all thread
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			for(int t=1; t<T; t++)
			{	
				d(t,k,i) = d(0,k,i);
			}
			dm(k,i) = d(0,k,i);
		}
	}
	
	// for(int k=0; k < K;k++)
	// {
	// 	cout << "D"<<k<<":\t";
	// 	for(int i=-dsize; i<=dsize; i++)
	// 	{
	// 		cout << d(0,k,i)<<"\t";
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;
}

void initBeta(int t)
{
	// init Beta = D*z
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			b(t,k,i) = 0;
//			cout << "At : "<< i << " \n ";
			for(int j =-dsize ; j <= dsize ;j++)
			{
				if(i+j <0 || i+j >=ssize) continue;
//				cout << "s "<<i+j<<"\t d "<<-j <<endl;
				b(t,k,i) -= s(t,i+j) * d(t,k,-j);
			}
//			cout << b(k,i)<<endl;
		}
	}
}

void inference(int t)
{
	// Coordinate Descent
	int round = 0;
	double lastLoss = 0;
	int lastK=-1;
	int lastI=-1;
	char x;
	while(true)
	{
		//calculate loss
		double loss =0 ;
		loss = calcLoss(t);
		// if(round%20==0)
		// {
		// 	cout<<"InferZ Loss("<<round<<"):\t";
		// 	printf("%.5f\n",loss);
		// }
		round++;

		if(fabs(lastLoss-loss) < 0.1)
		{
			// printf("end ! %.5f %.5f\n",loss,lastLoss);
			return;
		}
		if(round  > maxCoordinateDescentLoop ) break;
		if(loss==0) break;
//		scanf("%c",&x);

//		reportZ();
//		reportD();
		lastLoss = loss;

		// calculate new Z
		for(int k=0;k<K;k++)
		{
			for(int i= -dsize ; i< ssize + dsize; i++)
			{
				double sqdivider = 0;
				for(int j = -dsize; j<=dsize; j++)
				{
					if(i+j < 0 || i+j >= ssize) continue;
					sqdivider += (d(t,k,-j) * d(t,k,-j));
				}
				zn(t,k,i) = shrink(-b(t,k,i),LAMBDA) / sqdivider;
			}
		}
		//maximum diff Z
		double maxV=-1;
		int maxK;
		int maxI;

		for(int k =0 ; k < K ; k++)
		{
			for(int i=-dsize ; i < ssize + dsize; i++)
			{
				double fs = fabs(zn(t,k,i) - z(t,k,i) );
				if(fs > maxV  && lastK!=k && lastI!=i)
				{
					maxV=fs;
					maxK=k;
					maxI=i;
				}
			}
		}
//		cout << "maxV : " <<maxV ;
//		cout << "\tmaxK : " <<maxK ;
//		cout << "\tmaxI : " <<maxI ;
//		cout << "\tsig : " <<s(maxI) ;
//		cout << "\tnewZ : " << zn(maxK,maxI)<<endl;
//		reportB();
		// save old beta
		double savedBeta = b(t,maxK,maxI);

		// update beta
		for(int i=-2*dsize; i<= 2*dsize; i++)
		{
			int editing = i+maxI;
			if(editing < -dsize || editing >= ssize + dsize) continue;
			for(int k=0; k<K ;k++)
			{
				double total = 0;
				for(int r = -dsize + editing; r<=dsize + editing; r++)
				{
					if(r < 0 || r >= ssize) continue;
					for(int c = -dsize ; c<= dsize ;c++)
					{
						int zind = r+c;
						if( zind == maxI )
						{
							total += d(t,maxK,c) * d(t,k,-(r-editing));
//							cout << "sum : d("<<maxK<<","<<c<<")="<< d(maxK,c) <<" * d("<<k<<","<<-(r-editing)<<")"<<d(k,-(r-editing))<<"\n";
						}
					}
				}
				b(t,k,editing) += (zn(t,maxK,maxI)-z(t,maxK,maxI)) * total;
//				cout << "up : "<< k<<" "<<editing << " : " << b(k,editing) <<endl<<endl;
			}
		}

		// restore beta
//		if(round %10==0) cout << "change z : " << abs(z(maxK,maxI))<<endl;
		z(t,maxK,maxI) = zn(t,maxK,maxI);
		b(t,maxK,maxI) = savedBeta;
	}
	return;
}

void learnDictionary(int t)
{
	long double stepsize = 0.00001;
	long double lastLoss=99999999;
	int round = 0 ;
	while(round < MAXDICITER)
	{
		double loss = calcLoss(t);
		
		// cout << "Loss("<<round<<") :\t";
		// printf("%.6f\n",loss);
		round++;

		if(fabs(loss-lastLoss) < limitDiffLoss) return;
		if(loss<0.0001) return;
//		if(loss>lastLoss)
//		{
//			stepsize/=2;
//			cout<<"*";
//		}
//		else
//		{
//			stepsize+=stepsizeInc;
//		}
		lastLoss=loss;

		vector<long double> precalc(ssize,0);
		for(int i = 0 ; i < ssize; i++)
		{
			precalc[i] = s(t,i);
			for(int k =0 ; k < K ;k++)
			{
				for(int j=-dsize;j<=dsize;j++)
				{
					precalc[i] -= d(t,k,j) * z(t,k,i+j);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int d=-dsize;d<=dsize; d++)
			{
				nd(t,k,d) = 0;
				for(int i = 0 ; i < ssize;i++)
				{
					nd(t,k,d) += precalc[i] * -z(t,k,i+d);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int ds=-dsize;ds<=dsize; ds++)
			{
				if(nd(t,k,ds) > 1000) nd(t,k,ds) = 1000;
				if(nd(t,k,ds) < -1000) nd(t,k,ds) = -1000;
				d(t,k,ds) -=  stepsize * nd(t,k,ds);
			}
		}
	}
}

// ^^^^^^^^^^^^^^^^^^^^^^ OPTIMIZATION PART
//-----------------------
// vvvvvvvvvvvvvvvvvvvvvv FILE INPUT PART

vector< vector<double> > timeSeries(string path)
{
	ifstream myReadFile;
	
	myReadFile.open(path.c_str(),std::ifstream::binary);
	string line;
	if (myReadFile.is_open()) {
		
		int countII=0;
		int countV=0;
		int countABP=0;

		char sigName[32];
		long sigStart = 0;
		short freq = 0;
		short sigLen = 0;
		char hasII = 0;
		char hasV = 0;
		char hasABP = 0;
		char unuse = 0;
		double baseII;
		double gainII;
		double baseV;
		double gainV;
		double baseABP;
		double gainABP;

		int recordLength = 0;

		short shII[1250];
		short shV[1250];
		short shABP[1250];

		//fixed for fast !
		countII = 7260032;
		if(maxData != 0 )
			countII = maxData;
		// while (!myReadFile.eof())
  //   	{
		// 	myReadFile.read((char*)&recordLength, 4);

		// 	myReadFile.read((char*)&sigName, 32);
		// 	myReadFile.read((char*)&sigStart, 8);
		// 	myReadFile.read((char*)&freq, 2);
		// 	myReadFile.read((char*)&sigLen, 2);
			
		// 	myReadFile.read((char*)&hasII, 1);
		// 	myReadFile.read((char*)&hasV, 1);
		// 	myReadFile.read((char*)&hasABP, 1);
		// 	myReadFile.read((char*)&unuse, 1);
			
		// 	myReadFile.read((char*)&baseII, 8);
		// 	myReadFile.read((char*)&gainII, 8);
		// 	myReadFile.read((char*)&baseV, 8);
		// 	myReadFile.read((char*)&gainV, 8);
		// 	myReadFile.read((char*)&baseABP, 8);
		// 	myReadFile.read((char*)&gainABP, 8);

		// 	// printf("Read Byte : %d\n",recordLength);
		// 	// printf("Name : %s\n",sigName);
		// 	// printf("SigStart : %ld\n",sigStart);
		// 	// printf("Frequency : %d\n",freq);
		// 	// printf("SigLength : %d\n",sigLen);
		// 	// printf("has II : %d\n",hasII);
		// 	// printf("has V : %d\n",hasV);
		// 	// printf("has ABP : %d\n",hasABP);
		// 	// printf("BASE GAIN : %f %f %f %f %f %f \n\n",baseII,gainII,baseV,gainV,baseABP,gainABP);

		// 	if(hasII==1) countII++;
		// 	if(hasV==1) countV++;
		// 	if(hasABP==1) countABP++;
			
		// 	if(hasII==1) myReadFile.read((char*)&shII, 1250*2);
		// 	if(hasV==1) myReadFile.read((char*)&shV, 1250*2);
		// 	if(hasABP==1) myReadFile.read((char*)&shABP, 1250*2);

		// 	if(freq != 125)
		// 	{
		// 		printf("***FREQ!=125\n");
		// 	}
		// 	// break;
		// }
		// printf("II %d / V %d / ABP %d\n",countII,countV,countABP);

		myReadFile.seekg(0, ios::beg);

		vector< vector<double> > v(countII,vector<double>(1250,0));

		int i = 0;
		while (!myReadFile.eof())
		{
			myReadFile.read((char*)&recordLength, 4);

			myReadFile.read((char*)&sigName, 32);
			myReadFile.read((char*)&sigStart, 8);
			myReadFile.read((char*)&freq, 2);
			myReadFile.read((char*)&sigLen, 2);
			
			myReadFile.read((char*)&hasII, 1);
			myReadFile.read((char*)&hasV, 1);
			myReadFile.read((char*)&hasABP, 1);
			myReadFile.read((char*)&unuse, 1);
			
			myReadFile.read((char*)&baseII, 8);
			myReadFile.read((char*)&gainII, 8);
			myReadFile.read((char*)&baseV, 8);
			myReadFile.read((char*)&gainV, 8);
			myReadFile.read((char*)&baseABP, 8);
			myReadFile.read((char*)&gainABP, 8);

			// printf("Read Byte : %d\n",recordLength);
			// printf("Name : %s\n",sigName);
			// printf("SigStart : %ld\n",sigStart);
			// printf("Frequency : %d\n",freq);
			// printf("SigLength : %d\n",sigLen);
			// printf("has II : %d\n",hasII);
			// printf("has V : %d\n",hasV);
			// printf("has ABP : %d\n",hasABP);
			// printf("BASE GAIN : %f %f %f %f %f %f \n\n",baseII,gainII,baseV,gainV,baseABP,gainABP);

			if(hasII==1) myReadFile.read((char*)&shII, 1250*2);
			if(hasV==1) myReadFile.read((char*)&shV, 1250*2);
			if(hasABP==1) myReadFile.read((char*)&shABP, 1250*2);

			if(hasII==1)
			{
				for(int j=0; j< 1250;j++)
				{
					v[i][j] = (shII[j] - baseII) * gainII;
				}
				i++;
			}
			// break;
			if(i==countII) break;
		}
		printf("read file done");
		return v;
	}
	else
	{
		cout << "404 file not found ! :(" << endl;
		vector< vector<double> > v(0,vector<double>(0,0));
		return v;
	}
}

vector<int> shuffleArray(int total)
{
	vector<int> v(total,0);
	for(int i = 0 ; i < total ; i++)
	{
		v[i]=i;
	}
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    return v;
}

int main(void){
	ts = timeSeries("/home/kanit/npz/d-8634");

	// for(int i = 0; i < 1250;i++)
	// {
	// 	printf("%.4f,",ts[93939][i]);
	// }
	// return 0;
//	cout << ts.size()<<endl; return 0;
//	vector< vector<double> > ts(1,vector<double>(1250,0));
//	for(int j=0;j<ssize;j++) ts[0][j]=tss[j];

	if(importDict)
	{
		importDictionary();
	}
	else
	{
		initD();
	}

	double sumLoss = 0;
	int count = 0;
	int count0 = 0;
	int countPrint = 0;

	vector<int> rd = shuffleArray(ts.size());

	#pragma omp parallel for schedule(static) num_threads(T)
	for(int i = 0 ; i < ts.size() ; i++)
	{
		int t = omp_get_thread_num();

		initS(t,ts[rd[i]]);
		initZ(t);
		initBeta(t);
		// normalizeS(t);
		inference(t);
		learnDictionary(t);
		// normalizeDictionary(t);
		printf("-");
		fflush(0);
		
		#pragma omp critical
		{
			double loss = calcLoss(t);
			if(loss == loss)
			{
				sumLoss += loss;
				count++;
				syncDictionary(t);
				if(t==2) count0++;
				if(t==2 && count0 == miniBatch)
				{
					char logname[100];
					sprintf(logname,"res_%06d_%d",countPrint++,(int)(sumLoss/count));
					reportTesting(t,logname);
					printf("AvgLoss (%d): %f\n",count,sumLoss/count);
					count0 = 0 ;
					sumLoss = 0 ;
				}
			}
			else
			{
				printf("s");
			}
		}
	}
}

