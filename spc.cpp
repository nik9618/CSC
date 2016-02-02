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

#define T 30
#define K 40
#define dsize 4
#define ssize 1250
#define importDict true
using namespace std;

string resultpath = "/home/kanit/anomalydeep/result/";

double limitDiffLoss = 0.001;
int maxCoordinateDescentLoop = 1000;
int miniBatch = 5;
int maxData = 4000000;
double impDict[K*(dsize*2+1)] = {0.252214,0.254039,0.253566,0.25284,0.252187,0.250447,0.249278,0.246108,0.238956,0.199918,0.237786,0.260485,0.275722,0.279185,0.273421,0.261366,0.237154,0.20925,0.0346105,0.21491,0.482694,0.453023,0.256584,0.0600107,-0.0276764,-0.0333672,-0.0280695,0.238532,0.246959,0.248789,0.250051,0.250948,0.25224,0.253067,0.254114,0.254866,0.24676,0.250263,0.250698,0.250735,0.250618,0.250595,0.250446,0.250173,0.24946,0.232791,0.250106,0.256553,0.257917,0.257752,0.255434,0.252225,0.247434,0.23908,0.247139,0.505377,0.418016,0.205396,0.0731005,0.0350801,0.0460451,0.0494022,0.0456633,0.24727,0.250249,0.250578,0.250613,0.250567,0.250534,0.250393,0.250086,0.249496,-0.0194359,0.537391,0.493961,0.238155,0.0311902,-0.0534182,-0.0662996,-0.0375748,-0.0616146,0.0250429,0.348735,0.555104,0.3405,0.122304,-0.0868289,-0.0791268,-0.0505067,-0.0620842,0.246955,0.250247,0.250603,0.250675,0.250591,0.250579,0.250421,0.250153,0.249538,0.243332,0.268882,0.284059,0.284381,0.245995,0.240647,0.230458,0.22235,0.215469,0.0344575,0.0415095,0.0558933,0.0962489,0.273005,0.308653,0.326283,0.336586,0.340087,0.260426,0.259373,0.256867,0.254259,0.25167,0.248096,0.244743,0.240615,0.232541,-0.00222387,0.00826846,0.0410832,0.313846,0.390791,0.391531,0.34012,0.0929336,0.0283397,0.24736,0.250301,0.250474,0.250533,0.250465,0.250486,0.25038,0.250162,0.2496,0.230942,0.24114,0.245365,0.248403,0.251225,0.25413,0.256787,0.259454,0.26174,0.1516,0.160747,0.159433,0.157517,0.155401,0.241476,0.415349,0.414148,0.129676,0.240317,0.277722,0.283399,0.284349,0.279726,0.275487,0.268183,0.0784464,0.00629009,0.248509,0.250793,0.250878,0.250819,0.250714,0.250439,0.25019,0.249517,0.247945,0.0603019,-0.228123,-0.322328,0.320544,0.25649,0.28918,0.263579,0.193077,0.173145,0.273821,0.268474,0.26222,0.256116,0.25018,0.243875,0.237903,0.230982,0.221013,0.0349322,0.134194,0.387906,0.433854,0.379007,0.158601,0.0613494,0.0134148,0.0032677,0.00847514,0.0350537,0.0889643,0.181927,0.335067,0.407293,0.35603,0.243562,0.124696,0.234838,0.245054,0.247892,0.249642,0.250975,0.252598,0.25396,0.256148,0.258276,0.245627,0.249814,0.250413,0.250634,0.250678,0.250823,0.250784,0.250697,0.250279,0.247811,0.250273,0.250567,0.250408,0.250512,0.250359,0.250465,0.250055,0.24932,0.249249,0.251273,0.251286,0.251124,0.25093,0.250429,0.249997,0.24893,0.246619,0.211921,0.224607,0.234317,0.242814,0.251227,0.258858,0.266358,0.274123,0.281103,0.248164,0.250551,0.250582,0.250634,0.250567,0.250458,0.250253,0.249836,0.24873,-0.0053123,-0.0466071,0.0117558,-0.0547337,0.752709,0.329752,-0.0194489,-0.037739,-0.0365781,0.256683,0.256551,0.255293,0.253317,0.251863,0.249171,0.247259,0.243453,0.23565,0.265817,0.262775,0.258872,0.255062,0.25099,0.24649,0.242127,0.23712,0.228127,0.17346,0.194684,0.212624,0.230531,0.246899,0.266492,0.282601,0.300921,0.312039,0.116726,0.551168,0.507644,0.156104,-0.0419967,-0.0636136,-0.0556823,-0.0513175,-0.0487525,-0.0347689,-0.0487257,-0.0661784,-0.0485011,0.073547,0.40025,0.503163,0.373935,0.119771,0.240668,0.248706,0.249977,0.250674,0.251012,0.251705,0.251972,0.252389,0.252539,0.223368,0.235792,0.242033,0.246819,0.251034,0.255376,0.259747,0.264726,0.269573,0.295151,0.285632,0.273134,0.260714,0.248599,0.235897,0.22375,0.211152,0.196029,0.247472,0.25027,0.250613,0.250567,0.250511,0.250341,0.250336,0.249932,0.249724};

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
			for(int t=1; t<T; t++)
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
	#pragma omp parallel for schedule(static) num_threads(T)
	for(int i = 0 ; i < ts.size() ; i++)
	{
		int t = omp_get_thread_num();

		initS(t,ts[i]);
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

