import java.util.*;
import java.util.Scanner;


public class Solution {
	static int mod=1000000007;;
	
//	 
	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		System.out.println();

		Scanner sc=new Scanner(System.in);
		int tt=sc.nextInt();
//		
//		
		while(tt-->0){
			int n=sc.nextInt();
			
			int degree[]=new int[n];
			
			Arrays.fill(degree, 0);
			int u[]=new int[n-1];
			int v[]=new int[n-1];
			boolean brea=false;
			HashMap<Integer,LinkedList<Integer>>graph=new HashMap<>();
			for(int i=0;i<n-1;i++) {
				u[i]=sc.nextInt();
				v[i]=sc.nextInt();
				u[i]--;
				v[i]--;
				
				if(graph.containsKey(u[i])) {
					LinkedList<Integer> ll=graph.get(u[i]);
					ll.add(i);
					graph.put(u[i],ll);
				}
				else {
					LinkedList<Integer> ll=new LinkedList<>();
					ll.add(i);
					graph.put(u[i],ll);
				}
				if(graph.containsKey(v[i])) {
					LinkedList<Integer> ll=graph.get(v[i]);
					ll.add(i);
					graph.put(v[i],ll);
				}
				else {
					LinkedList<Integer> ll=new LinkedList<>();
					ll.add(i);
					graph.put(v[i],ll);
				}
				
				
				
				degree[u[i]]++;
				if(degree[u[i]]>2)brea=true;
				degree[v[i]]++;
				if(degree[v[i]]>2)brea=true;
			}
			
			if(brea) {
				System.out.println(-1);
				continue;
			}
			
			
			
			int i;
			for( i=0;i<n;i++) {
				
				if(degree[i]==1)break;
				
			}
//			System.out.println(graph+" "+i);
			
			int ans[]=new int[n-1];
			int ind=graph.get(i).get(0);
			ans[ind]=2;
			
			if(u[ind]==i)i=v[ind];
			else i=u[ind];
			
			boolean flip=true;
			
			while(graph.get(i).size()>1) {
				
				if(graph.get(i).get(0)==ind) {
					ind=graph.get(i).get(1);
				}
				else {
					ind=graph.get(i).get(0);
				}
				
				if(flip) {
					ans[ind]=3;
				}else ans[ind]=2;
				
				flip=!flip;
				
				
				if(u[ind]==i)i=v[ind];
				else i=u[ind];
				
			}
			
			
			 
			for(int j:ans)
			 System.out.print(j+" ");
			 System.out.println();
			
		
			
		}
	}
		
		
	
		
		
	
	
	


}
