import java.io.*;
import java.util.*;
public class E {


	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Reader sc=new Reader();
		PrintWriter pw=new PrintWriter(System.out);
		int t=sc.nextInt();
		while(t-->0) {
			int n=sc.nextInt();
			int k=sc.nextInt();
			int[] idxes=new int[k];
			int[] temps=new int[k];
			for(int i=0;i<k;i++) {
				idxes[i]=sc.nextInt()-1;
			}
			for(int i=0;i<k;i++) {
				temps[i]=sc.nextInt();
			}
			int[] ans=new int[n];
			Arrays.fill(ans, Integer.MAX_VALUE/2);
			for(int i=0;i<k;i++) {
				ans[idxes[i]]=temps[i];
			}
			for(int i=1;i<n;i++) {
				ans[i]=Math.min(ans[i], ans[i-1]+1);
			}
			for(int i=n-2;i>=0;i--) {
				ans[i]=Math.min(ans[i], ans[i+1]+1);
			}
			for(int i=0;i<n;i++) {
				pw.print(ans[i]+" ");
			}
			pw.println();
			
		}
		pw.flush();
		sc.close();
	}
	
	

}
