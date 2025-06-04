import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Main
{
	public static void main (String[] args) throws java.lang.Exception
	{
		// your code goes here
		Scanner sc=new Scanner(System.in);
		int t=sc.nextInt();
		PrintWriter out=new PrintWriter(System.out);
		while(t-->0) {
		    int n=sc.nextInt();
		    int freq[][]=new int[n][5];
		    int rem[][]=new int[n][5];
		    for(int i=0;i<n;i++) {
		        String str=sc.next();
		        
		        for(int j=0;j<str.length();j++) {
		            freq[i][str.charAt(j)-'a']++;
		        }
		        for(int k=0;k<5;k++) {
		            rem[i][k]=str.length()-freq[i][k];
		        }
		    }
		    
		    int ans=0;
		    for(int i=0;i<5;i++) {
		        int arr[]=new int[n];
		        for(int j=0;j<n;j++)
		        arr[j]=freq[j][i]-rem[j][i];
		        
		        Arrays.sort(arr);
		        int total=0;
		        int sum=0;
		        for(int k=n-1;k>=0;k--) {
		            if(sum+arr[k]>0) {
		                sum=sum+arr[k];
		                total++;
		            }
		            else {
		                break;
		            }
		        }
		        ans=Math.max(ans,total);
		    }
		    out.println(ans);
		    
		}
		out.flush();
		out.close();
	}
}
