import java.util.*;
import java.lang.*;
import java.io.*;
public class Main
{
    static void sort(int a[]){
        ArrayList<Integer> arr=new ArrayList<>();
        for(int i=0;i<a.length;i++)arr.add(a[i]);
        Collections.sort(arr);
        for(int i=0;i<a.length;i++)a[i]=arr.get(i);
    }
	public static void main (String[] args) throws java.lang.Exception
	{
	   // Scanner sc=new Scanner(System.in);
	    BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
	   // int t=1;
	    //t=sc.nextInt();
	    int t=Integer.parseInt(br.readLine());
	    while(--t>=0){
	        int n=Integer.parseInt(br.readLine());
	         int a[]=new int[n];
	         StringTokenizer st=new StringTokenizer(br.readLine());
	        for(int i=0;i<n;i++)a[i]=Integer.parseInt(st.nextToken());
	       String s=br.readLine();
	        ArrayList<Integer> inc=new ArrayList<>();
	        ArrayList<Integer> dec=new ArrayList<>();
	        
	        for(int i=0;i<n;i++){
	            if(s.charAt(i)=='R')inc.add(a[i]);
	            else dec.add(a[i]);
	        }
	        Collections.sort(dec);
	        Collections.sort(inc,Collections.reverseOrder());
	        int p=n;
	        boolean flag=false;
	        for(int i=0;i<inc.size();i++){
	            if(inc.get(i)>p)flag=true;
	            p--;
	        }
	        p=1;
	        for(int i=0;i<dec.size();i++){
	            if(dec.get(i)<p)flag=true;
	            p++;
	        }
	        if(flag)System.out.println("NO");
	        else System.out.println("YES");
	    }
	    
	}
}
