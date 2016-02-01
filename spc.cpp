//============================================================================
// Name        : AMHProject.cpp
// Author      : Nik
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define margin 100 // becareful margin must me at least 3*dsize
#define s(i) 		S[i+margin]
#define d(k,i) 		D[k][i+dsize]
#define nd(k,i) 	ND[k][i+dsize]
#define z(k,i) 		Z[k][i+margin]
#define zn(k,i) 	Zn[k][i+margin]
#define b(k,i) 		B[k][i+margin]
#define LAMBDA		0.1
#define MAXDICITER 	1

int dsize = 7;
int K = 50;
int ssize = 1250;
double limitDiffLoss = 0.001


using namespace std;

double shrink(double beta, double lamb)
{
	if(beta>lamb) return beta - lamb;
	if(beta<-lamb) return beta + lamb;
	return 0;
}

vector<double> S(ssize+margin,0);
vector< vector<double> > D(K,vector<double>(dsize*2+1,0));
vector<double> Zct(K,0);
vector< vector<double> > ND(K,vector<double>(dsize*2+1,0));
vector< vector<double> > Z(K,vector<double>(ssize+3*margin,0));
vector< vector<double> > Zn(K,vector<double>(ssize+3*margin,0));
vector< vector<double> > B(K,vector<double>(ssize+3*margin,0));
vector< vector<double> > ts;

double accumuZct()
{
	for(int k=0;k<K;k++)
	{
		double ct = 0.f;
		for(int i = -dsize; i<ssize+dsize ;i++)
		{
			ct += fabs(z(k,i));
		}
		Zct[k] = (0.99)*Zct[k] + (0.01) * ct;
		cout << "Zct["<<k<<"]" << Zct[k]<<endl;
	}
}

double calcLoss()
{
	double loss=0;
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(k,j) * z(k,i+j);
			}
		}
		loss+= x*x;
	}
//	if2
	return loss;
}

vector<double> reconstruct()
{
	vector<double> res(ssize,0);
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(k,j) * z(k,i+j);
			}
		}
		res[i] = x;
	}
//	if2
	return res;
}

void reportDZ()
{
	vector<double> res = reconstruct();
	for(int i =0; i< ssize;i++)
	{
		cout << res[i] <<"\t";
	}
	cout<<endl;
}

double lossDist()
{
	double loss=0;
	for(int i =0 ; i < ssize ;i++)
	{
		double x = s(i);
		for(int k = 0 ; k <  K ; k++)
		{
			for(int j =-dsize; j<= dsize ;j++)
			{
				x-= d(k,j) * z(k,i+j);
			}
		}
		cout<< fabs(x)<<"\t";
	}
//	if2
	return loss;
}

void reportZ()
{
	for(int k = 0 ; k < K ; k++)
	{
		cout << "Z"<<k<<":\t";
		for(int i = - dsize; i<= ssize + dsize ;i++)
		{
			cout << z(k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportD()
{
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = - dsize; i<= dsize ;i++)
		{
			cout << d(k,i) <<"\t";
		}
		cout<<endl;
	}
}

void reportB()
{
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			cout << b(k,i)<<"\t";
		}
		cout<<endl;
	}
}

void reportS() {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
//	for(int i = 0 ; i < ssize;i++)s(i) = in[i];
	for(int i = 0; i < ssize;i++) cout << s(i) <<"\t";
	cout<<endl<<endl;
}

void initS(vector<double> in) {
	//init signal
//	cout << "Signal("<< ssize <<") : ";
	for(int i = 0 ; i < ssize;i++)s(i) = in[i];
//	for(int i = -5 ; i < ssize+5 ;i++) cout << s(i) <<" ";
//	cout<<endl<<endl;
}

void initZ() {
	//init Z
	for(int k=0;k<K;k++)
	{
		for(int i = -dsize;i<ssize+dsize;i++)
		{
			z(k,i) = 0;
		}
	}
	cout<<endl<<endl;
}

void normalizeDictionary()
{
	for(int k = 0 ; k < K ; k++)
	{
		double total = 0.f;
		for(int i = -dsize; i <=dsize ; i++ )
		{
			total += d(k,i);
		}
		for(int i = -dsize; i <=dsize ; i++ )
		{
			d(k,i) /=total;
		}
	}
}

void initD()
{
	for(int k=0; k < K;k++)
	{
		for(int i=-dsize; i<=dsize; i++)
		{
			d(k,i) = 1./(1+2*dsize) + 1./(1+2*dsize)*(rand()%10000)/100000.0;
		}
	}
	cout << endl;
	normalizeDictionary();
	for(int k=0; k < K;k++)
	{
		cout << "D"<<k<<":\t";
		for(int i=-dsize; i<=dsize; i++)
		{
			cout << d(k,i)<<"\t";
		}
		cout << endl;
	}
	cout << endl;
}

void initBeta()
{
	// init Beta = D*z
	for(int k=0;k<K;k++)
	{
		for(int i = - dsize; i< ssize + dsize; i++)
		{
			b(k,i) = 0;
//			cout << "At : "<< i << " \n ";
			for(int j =-dsize ; j <= dsize ;j++)
			{
				if(i+j <0 || i+j >=ssize) continue;
//				cout << "s "<<i+j<<"\t d "<<-j <<endl;
				b(k,i) -= s(i+j) * d(k,-j);
			}
//			cout << b(k,i)<<endl;
		}
	}
}

void inference()
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
		loss = calcLoss();
		if(round%20==0)
		{
			cout<<"InferZ Loss("<<round<<"):\t";
			printf("%.5f\n",loss);
		}
		round++;

		if(fabs(lastLoss-loss) < 0.1)
		{
			printf("end ! %.5f %.5f\n",loss,lastLoss);
			return;
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
					sqdivider += (d(k,-j) * d(k,-j));
				}
				zn(k,i) = shrink(-b(k,i),LAMBDA) / sqdivider;
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
				double fs = fabs(zn(k,i) - z(k,i) );
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
		double savedBeta = b(maxK,maxI);

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
							total += d(maxK,c) * d(k,-(r-editing));
//							cout << "sum : d("<<maxK<<","<<c<<")="<< d(maxK,c) <<" * d("<<k<<","<<-(r-editing)<<")"<<d(k,-(r-editing))<<"\n";
						}
					}
				}
				b(k,editing) += (zn(maxK,maxI)-z(maxK,maxI)) * total;
//				cout << "up : "<< k<<" "<<editing << " : " << b(k,editing) <<endl<<endl;
			}
		}

		// restore beta
//		if(round %10==0) cout << "change z : " << abs(z(maxK,maxI))<<endl;
		z(maxK,maxI) = zn(maxK,maxI);
		b(maxK,maxI) = savedBeta;
	}
	return;
}

void learnDictionary()
{
	long double stepsize = 0.00001;
	long double lastLoss=99999999;
	int round = 0 ;
	while(round < MAXDICITER)
	{
		double loss = calcLoss();
		
		cout << "Loss("<<round<<") :\t";
		printf("%.6f\n",loss);
		round++;

		if(fabs(loss-lastLoss) < limitDiffLoss) return;
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
			precalc[i] = s(i);
			for(int k =0 ; k < K ;k++)
			{
				for(int j=-dsize;j<=dsize;j++)
				{
					precalc[i] -= d(k,j) * z(k,i+j);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int d=-dsize;d<=dsize; d++)
			{
				nd(k,d) = 0;
				for(int i = 0 ; i < ssize;i++)
				{
					nd(k,d) += precalc[i] * -z(k,i+d);
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int ds=-dsize;ds<=dsize; ds++)
			{
				d(k,ds) -=  stepsize * nd(k,ds);
			}
		}
	}
}

// ^^^^^^^^^^^^^^^^^^^^^^ OPTIMIZATION PART
//-----------------------
// vvvvvvvvvvvvvvvvvvvvvv FILE INPUT PART

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

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
		// countII = 7260032;
		countII = 100000;
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
		// printf("done");
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

	initD();
	int r = 0;
	while(r<ts.size(){
		initS(ts[r]);
		initZ();
		initBeta();
		cout << "Start Loss( "<<r <<" ): " << calcLoss()<<endl;
		inference();
//		reportZ();
		learnDictionary();
		normalizeDictionary();
//		cout<<"D"<<endl;
//		reportD();
//		cout<<"S"<<endl;
//		reportS();
//		cout<<"DZ"<<endl;
		reportDZ();
		lossDist();
//		accumuZct();
		cout << "End Loss( "<<r <<" ): " << calcLoss()<<endl;
//		break;
//		scanf(	"%c",&x);
		r++;
	}
	// ---- done dictionary optimization -----


	// ---- generate features----

	cout << "Start producing result" << endl;

	ofstream f;
	f.open ("results.txt");

	f << "#Information" << endl;
	f << "SampleSize\t" << ssize<<endl;
	f << "NSameple\t" << ts.size() <<endl;
	f << "Nfilter\t" << K <<endl;
	f << "FilterSize\t" << dsize*2+1<<endl;


	f << "#Dictionary" <<endl;
	for(int k = 0 ; k < K ; k++)
	{
		for(int i = - dsize; i<= dsize ;i++)
		{
			f << d(k,i) ;
			if(i!=dsize) f<<",";
		}
		f<<endl;
	}
	r=0;
	while(r<ts.size()-10){
		initS(ts[r]);
		initZ();
		initBeta();
		inference();

		f << "#Samples "<< r << endl;
		cout << "#Samples "<< r << endl;
		//report S
		f << "S"<<",";
		for(int i = 0; i < ssize;i++)
		{
			f<< s(i) ;
			if(i<ssize-1) f<< ",";
		}
		f<<endl;

		//report reconstruct
		vector<double> d = reconstruct();
		f << "DZ"<<",";
		for(int i = 0; i < ssize;i++)
		{
			f << d[i];
			if(i<ssize-1) f<< ",";
		}
		f<<endl;

		//report features
		for(int k = 0 ; k < K ; k++)
		{
			f << "Z"<<",";
			for(int i = - dsize; i<= ssize + dsize ;i++)
			{
				f << z(k,i);
				if(i<ssize + dsize)  f<< ",";
			}
			f<<endl;
		}
		r++;
	}
	f.close();
}

//int main(void)
//{
//	char x ;
//	int r = 1;
//	initD();
//
//	while(true)
//	{
//		r++;
//		vector<double> ts(ssize,2);
//		for(int i =0  ; i < 15; i++)
//		{
//			ts[30+i]=2+i;
//		}
//		for(int i = 15;i<30;i++)
//		{
//			ts[30+i]=2+30-i;
//		}
////		for(int i = 0;i<ssize;i++)
////		{
////			ts[i]/=40.;
////		}
//		initS(ts);
//		initZ();
//		cout << "Start Loss( "<<r <<" ): " << calcLoss()<<endl;
//
//		initBeta();
//		inference();
////		reportZ();
//		learnDictionary();
//		normalizeDictionary();
//		reportD();
//		cout << "End Loss( "<<r <<" ): " << calcLoss()<<endl;
//		if(r%100==0)scanf("%c",&x);
//	}
//}

//	for(int i =0 ; i < ts.size() ; i++)
//	{
//		for(int j = 0 ;j < SAMPLESIZE ;j++)
//		{
//			cout << ts[i][j]<<"," ;
//		}
//		cout << endl;
//	}
//	cout << "loading data "<< endl;
////	vector< vector<double> > ts = timeSeries("/Users/nik9618/Desktop/ubt/a40024.txt","II");
//	cout << "done loading data "<< endl;

