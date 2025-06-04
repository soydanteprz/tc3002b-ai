import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
		// your code goes here
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int t = Integer.parseInt(br.readLine());
		for(int q=0;q<t;q++){
		    //  int n = Integer.parseInt(br.readLine());
		      String s = br.readLine();
                  String s1[] = br.readLine().split(" ");
                  int n = Integer.parseInt(s1[0]);
                  int k = Integer.parseInt(s1[1]);
                  String s2[] = br.readLine().split(" ");
                  int a[] = new int[k];
                  for(int i=0;i<k;i++){
                        a[i] = Integer.parseInt(s2[i]);
                  }
                  String s3[] = br.readLine().split(" ");
                  int b[] = new int[k];
                  for(int i=0;i<k;i++){
                        b[i] = Integer.parseInt(s3[i]);
                  }
                  long ans[] = new long[n];
                  for(int i=0;i<n;i++){
                        ans[i] = Integer.MAX_VALUE;
                  }
                  for(int i=0;i<k;i++){
                        ans[a[i]-1] = b[i];
                  }
                  for(int i=1;i<n;i++){
                        ans[i] = Math.min(ans[i],ans[i-1]+1);
                  }
                  for(int i=n-2;i>=0;i--){
                        ans[i] = Math.min(ans[i],ans[i+1]+1);
                  }
                  
                  for(int i=0;i<n;i++){
                        System.out.print(ans[i]+" ");
                  }
                  System.out.println();
		}
	}
}
