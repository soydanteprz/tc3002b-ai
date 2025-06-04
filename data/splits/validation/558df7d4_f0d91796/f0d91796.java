import java.io.*;
import java.util.*;
 
public class cf {
 
    public static void main(String[] args){
 
        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while(t-- > 0){
 			
 			int n=sc.nextInt();
            int arr[]=new int[n];
            for(int i=0;i<n;i++){
                arr[i]=sc.nextInt();
            }
            int ans[]=new int[n];
            if(n%2==0){
                for(int i=0;i<n;i=i+2){
                    ans[i]=-arr[i+1];
                    ans[i+1]=arr[i];
                }
            }
            else{
                if(arr[0]+arr[1]!=0){
                    ans[0]=-arr[2];
                    ans[1]=-arr[2];
                    ans[2]=arr[0]+arr[1];
                }
                else{
                    if(arr[1]+arr[2]!=0){
                        ans[1]=-arr[0];
                        ans[2]=-arr[0];
                        ans[0]=arr[1]+arr[2];
                    }
                    else{
                        ans[0]=-arr[1];
                        ans[2]=-arr[1];
                        ans[1]=arr[0]+arr[2];
                    }
                }
                for(int i=3;i<n;i=i+2){
                    ans[i]=-arr[i+1];
                    ans[i+1]=arr[i];
                }
            }
            for(int j=0;j<n;j++){
                System.out.print(ans[j]+" ");
            }
            System.out.println();
            

            
            

            

            
           
 			

        }	        	
        	
 			
    }
 
    
    
    
 
}

//////////////////////////////////////////////////////////////
// LCM AND GCD
/*
	public static int gcd(int a,int b){
		if(b == 0){
			return a;
		}
		return gcd(b,a%b);
	}
 
	public static int lcm(int a,int b){
		 return (a / gcd(a, b)) * b;
	}*/


///////////////////////////////////////////////////////////////////////////////////

//Iterator



/*Iterator iterator = object.iterator();
 
        while (iterator.hasNext()) {
            System.out.print(iterator.next() + " ");
        }*/
///////////////////////////////////////////////////////////////////////////////////



