import java.io.*;

import java.util.*;
/*




*/

 
 public class A{
	static FastReader sc=null;
	
	public static void main(String[] args) {
		sc=new FastReader();
		int t=sc.nextInt();
		
		for(int tt=0;tt<t;tt++) {
			int n=sc.nextInt();
			char a[]=sc.next().toCharArray(),b[]=sc.next().toCharArray();
			
			int fa=0,fb=0,da=0,db=0,sum=0;
			boolean dif=false;
			for(int i=0;i<n;i++) {
				sum+=a[i]-'0';
				if(a[i]!=b[i]) {
					dif=true;
					if(a[i]=='1')fa++;
					else fb++;
				}
				else {
					if(a[i]=='1')da++;
					else db++;
				}
			}
			
			if(sum==0) {
				System.out.println(dif?-1:0);
				continue;
			}
			
			int ans=n+1;
			
			if(fa==fb) ans=(fa+fb);
			if(da==db+1) ans=Math.min(da+db, ans);
			
			System.out.println(ans==(n+1)?-1:ans);
		}
	}
	
	
	
	
	
	
	
	
}
