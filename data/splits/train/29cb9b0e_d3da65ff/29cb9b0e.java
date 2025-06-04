import java.util.*;
public class Main
{
  
    public static void solve (int ar[],int n)
    {
       if(n%2==1)
        {
    
           
            int c=ar[2];
            int b=ar[1];
            int a=ar[0];
            if(a+b!=0)
            {
               ar[2]=-(a+b);
               ar[1]=c;
               ar[0]=c;
            }
            else if(b+c!=0)
            {
                 ar[2]=a;
               ar[1]=a;
               ar[0]=-(b+c);
            }
            else
            {
                 ar[2]=b;
               ar[1]=-(a+c);
               ar[0]=b;
            }
            System.out.print(ar[0]+" "+ar[1]+" "+ar[2]+" ");
           for(int i=3;i<(n);i=i+2)
           {
               System.out.print((-ar[i+1])+" "+ar[i]+" ");
           } 
            System.out.println();
        }
          //  System.out.println(ans[n-1]+ " x"+ans[n-2]+" x"+ans[n-3]);
           
          else
          {
              for(int i=0;i<(n);i=i+2)
           {
               System.out.print((-ar[i+1])+" "+ar[i]+" ");
           }  
            System.out.println();
          }
          
           
        
    }
	public static void main(String[] args) {
		Scanner sc=new Scanner(System.in);int t=sc.nextInt();
		while(t-->0)
		{
		    int n=sc.nextInt();
		   int ar[]=new int[n];
		   for(int i=0;i<n;i++)
		   ar[i]=sc.nextInt();
		   solve(ar,n);
		}
	}
}
