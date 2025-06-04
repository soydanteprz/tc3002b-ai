import java.util.*;
public class MyClass {
    public static void main(String args[]) {
      Scanner sc=new Scanner(System.in);
      int tc=sc.nextInt();
      while(tc>0)
      {
          int n=sc.nextInt();
          int arr[]=new int[n];
          int i,max=Integer.MIN_VALUE,min=Integer.MAX_VALUE;
          for(i=0;i<n;i++)
          {
              arr[i]=sc.nextInt();
              if(arr[i]>max)
              max=arr[i];
              if(arr[i]<min)
              min=arr[i];
          }
          int low=min,high=max,mid,ans=min;
          while(low<=high)
          {
              mid=(low+high)/2;
              int temp[]=new int[n];
              for(i=0;i<n;i++)
              temp[i]=arr[i];
              boolean flag=true;
              for(i=n-1;i>=2;--i)
              {
                  if(temp[i]<mid)
                  {
                  flag=false;
                  break;
                  }
                  int deli=Math.min(temp[i]-mid,arr[i]);
                  int ex=deli/3;
                  temp[i]=temp[i]-(3*ex);
                  temp[i-1]=temp[i-1]+(ex);
                   temp[i-2]=temp[i-2]+(2*ex);
                   
              }
              
              if(temp[0]<mid || temp[1]<mid)
              flag=false;
              
              if(flag==true)
              {
              ans=mid;
              low=mid+1;
              }
              else
              high=mid-1;
          }
          System.out.println(ans);
          --tc;
      }}}
