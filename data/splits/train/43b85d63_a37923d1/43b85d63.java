import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Map.Entry;

import java.util.Random;
import java.util.TreeSet;

public final class CF_724_D2_D2
{


	static boolean verb=true;
	
	
	
	
	
	

	
	
	static void output(Object o){outputWln(""+o+"\n");	}
	
	static void outputWln(Object o){try {out.write(""+ o);} catch (Exception e) {}}


	

	

	



	static long mod=1000000007;



	// Global vars
	static BufferedWriter out;
	static InputReader reader;

	
	
	

	
	static int PX=Integer.MAX_VALUE;
	static int NX=Integer.MIN_VALUE;

	static void process() throws Exception {


		out = new BufferedWriter(new OutputStreamWriter(System.out));
		reader = new InputReader(System.in);

		int T=reader.readInt();
		int CX=26;
		for (int t=0;t<T;t++) {
		
		
			int n=reader.readInt();
		
			int[] b=new int[n];
			for (int i=0;i<n;i++) {
				b[i]=reader.readInt();
			}
			TreeSet<Integer> ts=new TreeSet<Integer>();
			ts.add(b[0]);
			ts.add(NX);
			ts.add(PX);
			int core=b[0];
			boolean ok=true;
			for (int i=1;i<n;i++) {
				int x=b[i];
				if (x==core) {
					//ok
				} else if (x<core) {
					int y=ts.lower(core);
					if (y>x) {
						ok=false;
						//log("fail 1");

						break;
					}
					//log("core:"+core+" x:"+x+" y:"+y);
					core=x;
					ts.add(x);
						
					
				} else {
					int y=ts.higher(core);
					if (y<x) {
						ok=false;
						//log("fail 2");
						break;
						
					}
					//log("core:"+core+" x:"+x+" y:"+y);
					core=x;
					ts.add(x);
					
				}
			}
			
			
			if (ok)
				output("YES");
			else
				output("NO");
			
			
		
		}


		try {
			out.close();
		} catch (Exception Ex) {
		}

	}


	public static void main(String[] args) throws Exception {
		process();

	}

	

}
