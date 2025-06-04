import java.io.*;

import java.util.*;
/*




*/

 
 public class C{
	static FastReader fs=null;
	static Scanner sc=null;
	
	public static void main(String[] args) {
		sc=new Scanner(System.in);
		int t=sc.nextInt();
		
		for(int tt=0;tt<t;tt++) {
			int n=sc.nextInt();
			
			boolean visited[]=new boolean[n+1];
			int p[]=new int[n+1];
			for(int i=1;i<=n;i++) {
				if(visited[i])continue;
				
				while(true) {
					int val=Query(i);
					if(val==i)break;
				}
				
				int cur=i;
				while(!visited[cur]) {
					int val=Query(i);
					visited[cur]=true;
					p[cur]=val;
					cur=val;
				}
			}
			
			System.out.print("! ");
			for(int i=1;i<=n;i++)System.out.print(p[i]+" ");
			System.out.println();
		}
	}
	static int Query(int id) {
		System.out.println("? "+id);
		int val=sc.nextInt();
		return val;
	}
	
	
	
	static void print(int a[]) {
		for(int e:a) {
			System.out.print(e+" ");
		}
		System.out.println();
	}
	
	
	
	
}
