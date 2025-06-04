import java.io.BufferedReader; 
import java.io.IOException; 
import java.io.InputStreamReader; 
import java.util.Scanner; 
import java.util.Random;
import java.util.Arrays;
import java.util.StringTokenizer; 
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Collections;
import java.util.*;
public class C 
{ 
    public static void main(String[] args) 
	{ 
		FastReader sc=new FastReader(); 
		StringBuffer ans=new StringBuffer();
		int test=sc.nextInt();
		while(test-->0)
		{
			int n=sc.nextInt();
			long arr[]=new long[n];
			for(int i=0;i<n;i++) {
				arr[i]=sc.nextLong();
			}
			long nn=n;
			long ep=1;
			long op=1;
			
			long se=arr[0];
			long so=arr[1];
			
			long sume=se;long sumo=so;
			long minm = (se*nn) +(so*nn);
			
			for(int i=2;i<n;i++) {
				if(i%2 == 0) {
					++ep;
					sume+=arr[i];
					se=Math.min(se, arr[i]);
				}
				else {
					++op;
					sumo+=arr[i];
					so=Math.min(so, arr[i]);
				}
				
				long cost = (sume)+(se*(nn-ep));
				cost+=((sumo)+(so*(nn-op)));
				minm=Math.min(cost, minm);
			}
			ans.append(minm+"\n");
		
		}
		
	    System.out.print(ans);
	} 
        
    static final Random random=new Random();
    
	
	
	
	
	 


}
