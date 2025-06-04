import java.util.*;
import java.lang.*;
import java.io.*;
public class Main
{
    static int func(char a[],char b[]){
        int n=a.length;
        int a1=0,b1=0;
        for(int i=0;i<n;i++){
            if(a[i]=='1')a1++;
            if(b[i]=='1')b1++;
        }
        if(a1!=b1)return 100000000;
        int cnt=0;
        for(int i=0;i<n;i++){
            if(a[i]=='1'&&b[i]!='1')cnt++;
        }
        return cnt*2;
    }
	public static void main (String[] args) throws java.lang.Exception
	{
	    BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
	    //int t=1;
	   int t=Integer.parseInt(br.readLine());
	    while(--t>=0){
	           int n=Integer.parseInt(br.readLine());
	           char a[]=br.readLine().toCharArray();
	          char b[]=br.readLine().toCharArray();
	         int x=func(a,b);
	         int ind=-1;
	         for(int i=0;i<n;i++){
	             if(a[i]==b[i]&&a[i]=='1'){
	                 ind=i;
	                 break;
	             }
	         }
	         int y=100000000;
	         if(ind>=0){
	             for(int i=0;i<n;i++){
	                 if(i==ind)continue;
	                 if(a[i]=='1')a[i]='0';
	                 else a[i]='1';
	             }
	             y=func(a,b)+1;
	         }
	         if(x>=1000000&&y>=1000000){
	             System.out.println(-1);
	         }
	         else
	        System.out.println(Math.min(x,y));
	     
	        
	    }
	    
	}
}
