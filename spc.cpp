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
#include <iomanip>


#define margin 100 // becareful margin must me at least 3*dsize
#define os(t,i) 	OS[t][i+margin]
#define s(t,i) 		S[t][i+margin]
#define d(t,k,i) 	D[t][k][i+dsize]
#define dm(k,i) 	DMASTER[k][i+dsize]
#define nd(t,k,i) 	ND[t][k][i+dsize]
#define z(t,k,i) 	Z[t][k][i+margin]
#define zn(t,k,i) 	Zn[t][k][i+margin]
#define b(t,k,i) 	B[t][k][i+margin]
#define LAMBDA		0.005//0.5
#define MAXDICITER 	10

#define T 28
#define K 40
#define dsize 4
#define ssize 1250
#define importDict false
using namespace std;

string resultpath = "/home/kanit/anomalydeep/result/";

double minLossDict = 0.001;
double limitDiffLossDict = 0.001;
long double dictstepsize = 0.00005;

int inferenceMinRound = 100;
int maxCoordinateDescentLoop = 5000;
double diffLossSPC = 0.01;
int miniBatch = 10;
int maxData = 0;
double impDict[K*(dsize*2+1)] = {0.385239,0.383891,0.382022,0.379933,0.377974,0.374771,0.37133,0.368804,0.357938,0.359548,0.377672,0.386229,0.388585,0.386957,0.381569,0.374409,0.368354,0.357761,0.0609023,0.313107,0.69394,0.65423,0.353204,0.0721431,-0.0409667,-0.0373537,-0.0267031,0.353332,0.364247,0.368915,0.372518,0.376529,0.38063,0.384017,0.388517,0.392516,0.34776,0.369283,0.383548,0.391809,0.395508,0.391544,0.381365,0.368431,0.349184,0.265933,0.342746,0.397052,0.435545,0.447405,0.430919,0.392436,0.337647,0.268711,0.423715,0.483659,0.465451,0.41387,0.348893,0.326113,0.289919,0.273818,0.258492,0.372976,0.376893,0.377012,0.377268,0.377016,0.377199,0.376051,0.376091,0.372612,0.0592267,0.7552,0.700799,0.211196,-0.0732461,-0.079778,-0.0773341,-0.0765089,-0.0755332,0.130276,0.467295,0.73176,0.549638,0.185048,-0.065576,-0.104471,-0.073164,-0.0536737,0.371064,0.376703,0.37681,0.376974,0.377091,0.37744,0.376355,0.376434,0.374221,0.351224,0.385489,0.402026,0.403145,0.397389,0.38544,0.370296,0.353262,0.325242,0.0383381,0.0500582,0.0981968,0.158733,0.414357,0.455025,0.477025,0.495469,0.507044,0.395891,0.39104,0.386262,0.381826,0.376951,0.371775,0.366041,0.36067,0.349588,0.00732822,0.012672,0.084819,0.502024,0.56644,0.558297,0.486301,0.101958,0.0272025,0.370176,0.376796,0.377104,0.377291,0.377124,0.377278,0.376256,0.376494,0.374598,0.311063,0.329146,0.343932,0.35882,0.374651,0.391528,0.406187,0.419669,0.429446,0.346174,0.356545,0.36352,0.370023,0.376466,0.383059,0.388914,0.394939,0.399637,0.424694,0.417284,0.413599,0.41686,0.406015,0.40928,0.404321,0.0420922,0.0234369,0.379917,0.380561,0.380679,0.379999,0.379405,0.377582,0.374672,0.371694,0.35763,0.0252473,0.0366114,0.0958673,0.379248,0.484916,0.521777,0.493627,0.41526,0.277322,0.421681,0.402789,0.391787,0.380783,0.373006,0.363916,0.355964,0.34743,0.33666,-0.00397451,0.120889,0.602348,0.641233,0.560958,0.128916,0.0319628,0.0275104,0.0252267,0.335902,0.360466,0.374221,0.386361,0.393667,0.397714,0.39277,0.381879,0.35344,0.333209,0.349092,0.358201,0.365437,0.374043,0.38238,0.391737,0.402548,0.418504,0.361819,0.371447,0.374039,0.375774,0.377552,0.379249,0.380071,0.381278,0.381351,0.371708,0.376557,0.3774,0.376844,0.377574,0.376894,0.377122,0.375732,0.373316,0.385213,0.389985,0.378409,0.374289,0.372354,0.374862,0.374471,0.376009,0.356085,0.0628071,0.357635,0.372329,0.383724,0.391945,0.399803,0.405311,0.409261,0.404719,0.372641,0.37819,0.378915,0.379062,0.3787,0.37834,0.376266,0.37514,0.365529,-0.0602659,-0.0494533,-0.0507232,-0.113832,1.05947,0.0846124,-0.033103,-0.0102619,-0.0260999,0.390118,0.387303,0.384266,0.380605,0.377229,0.373278,0.369162,0.364844,0.354375,0.401625,0.39839,0.390894,0.383571,0.376452,0.369386,0.361569,0.354287,0.341581,0.366388,0.374197,0.378482,0.38082,0.382174,0.38136,0.378637,0.374561,0.365852,0.074366,0.683906,0.684329,-0.109003,-0.161608,-0.159193,-0.172074,-0.157146,-0.221698,-0.187795,-0.275477,-0.314907,-0.309035,-0.157679,0.518076,0.652801,0.391172,0.15974,0.357879,0.369155,0.372178,0.374626,0.377004,0.379302,0.38084,0.384007,0.387137,0.301944,0.308872,0.320209,0.334086,0.352218,0.38193,0.412741,0.453592,0.476161,0.431189,0.419132,0.403208,0.388561,0.374969,0.36101,0.346309,0.330761,0.307442,0.363527,0.375529,0.377288,0.378077,0.378594,0.379318,0.378105,0.378093,0.374165};

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
		double x = 0;
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x += d(t,k,j) * z(t,k,i+j);
			}
		}
		res[i] = x;
	}
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
	int morezero =0;
	int all = 0;
	for(int k =0 ; k < K ; k++)
	{
		for(int i = -dsize ; i < ssize+dsize;i++)
		{
			if(z(t,k,i)!=0)
			{
				morezero++;
			}
			all++;
		}
	}
	myfile << "sparsity="<<morezero<<"/"<<all<<"\n\n";
	myfile << "S" << "=[";	
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << s(t,i)<<" ";
	}
	myfile << "]\n\n";

	vector<double> rec = reconstruct(t);
	myfile << "R" << "=[";	
	for(int i = 0 ; i < ssize ; i++)
	{
		myfile << rec[i]<<" ";
	}
	myfile << "]\n\n";
	
	// for(int k=0;k<K;k++)
	// {
	// 	myfile << "Z" << k << "=["<<setprecision(3);	
	// 	for(int i=-dsize;i<ssize+dsize; i++)
	// 	{
	// 		myfile << z(t,k,i) <<" ";
	// 	}
	// 	myfile << "]\n";
	// 	break;
	// }

	myfile << "D" << "=["<<setprecision(10) ;
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
		// s(t,i) = ((s(t,i)-mean)/std);
		s(t,i) = ((s(t,i)-mean));
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
		for(int k = 0 ; k < K ; k++)
		{
			for(int i = -dsize; i <=dsize ; i++ )
			{
		
				dm(k,i) = 0.9*dm(k,i) + 0.1*d(t,k,i);
				d(t,k,i) = dm(k,i);
			}
		}
	}
	else
	{
		printf("NAN\n");
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
			d(0,k,i) = 1./(1+2*dsize) + 1./(1+2*dsize)*(rand()%10000)/1000000.0;
			// d(0,k,i) = 1./(1+2*dsize)*(rand()%10000)/1000000.0;
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

int inference(int t)
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

		if(round > inferenceMinRound )
		{
			if(fabs(lastLoss-loss) < diffLossSPC)
			{
				// printf("end ! %.5f %d\n",loss,round);
				break;
			}
			if(round  > maxCoordinateDescentLoop ) break;
			if(loss==0) break;
		}
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
	return round;
}

void learnDictionary(int t)
{
	long double lastLoss=99999999;
	int round = 0 ;
	while(round < MAXDICITER)
	{
		double loss = calcLoss(t);
		
		// cout << "DLoss("<<round<<") :\t";
		// printf("%.6f\n",loss);
		round++;

		if(fabs(loss-lastLoss) < limitDiffLossDict) return;
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
				d(t,k,ds) -=  dictstepsize * nd(t,k,ds);
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
	int countLoss = 0;
	int sumRound = 0;
	int count0 = 0;
	int countPrint = 0;
	int count = 0;
	vector<int> rd = shuffleArray(ts.size());

	#pragma omp parallel for num_threads(T)
	for(int i = 0 ; i < ts.size() ; i++)
	{
		int t = omp_get_thread_num();

		initS(t,ts[rd[i]]);
		initZ(t);
		initBeta(t);
		normalizeS(t);
		int inferRound = inference(t);
		double loss = calcLoss(t);
		learnDictionary(t);
		// normalizeDictionary(t);
		// printf("-");
		fflush(0);
		
		#pragma omp critical(CRIT_1)
	{
			sumLoss += loss;
			countLoss++;
			sumRound += inferRound;
			count++;
			syncDictionary(t);
			if(t==0) count0++;
			if(t==0 && count0 == miniBatch)
			{
				char logname[100];
				sprintf(logname,"res_%06d_%d",countPrint++,(int)(sumLoss/countLoss));
				reportTesting(t,logname);
				printf("AvgLoss (%d - %d): %f (%d)\n",count,countLoss,sumLoss/countLoss,sumRound/countLoss);
				countLoss = 0 ;
				count0 = 0 ;
				sumLoss = 0 ;
				sumRound = 0 ;
			}
		}
	}
}

